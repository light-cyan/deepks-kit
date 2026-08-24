from collections.abc import Mapping
import hashlib
from types import MappingProxyType
from weakref import WeakSet
import os,time,sys
import numpy as np
import torch

from deepks.data.force_schema import (
    ForceDataError,
    SCHEMA_FILENAME,
    load_force_dataset,
)


FORCE_MODE_NONE = "none"
FORCE_MODE_DEEPHF_RELAXED = "deephf_relaxed"
FORCE_DATA_MODES = {FORCE_MODE_NONE, FORCE_MODE_DEEPHF_RELAXED}


_FORCE_BATCH_ISSUERS = WeakSet()


def _tensor_fingerprint(value) -> bytes:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.digest()


class _ForceBatchIssuer:
    __slots__ = ("contract", "fingerprints", "frame_count", "__weakref__")

    def __new__(cls, *args, **kwargs):
        raise TypeError("force-batch issuers are created only by validated readers")

    def __setattr__(self, name, value):
        raise AttributeError("force-batch issuers are immutable")


class _ForceBatch(Mapping):
    """Immutable reader-issued tensors bound to frame selections and content."""

    __slots__ = ("_values", "_selections")

    def __new__(cls, *args, **kwargs):
        raise TypeError("force batches are issued only by validated readers")

    @classmethod
    def _from_issued(cls, values, selections):
        values = dict(values)
        selections = tuple(
            (issuer, tuple(indices)) for issuer, indices in selections
        )
        if any(issuer not in _FORCE_BATCH_ISSUERS for issuer, _indices in selections):
            raise TypeError("force batches require registered reader issuers")
        frame_count = next(iter(values.values())).shape[0]
        if any(value.shape[0] != frame_count for value in values.values()):
            raise ValueError("force-batch fields must share one frame axis")
        if sum(len(indices) for _issuer, indices in selections) != frame_count:
            raise ValueError("force-batch frame selections do not match tensor rows")
        if any(
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or index < 0
            or index >= issuer.frame_count
            for issuer, indices in selections
            for index in indices
        ):
            raise ValueError("force-batch frame selections are invalid")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_values", MappingProxyType(values))
        object.__setattr__(instance, "_selections", selections)
        return instance

    def __setattr__(self, name, value):
        raise AttributeError("force batches are immutable")

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _force_batch_error(batch, accepted_contracts) -> str | None:
    if type(batch) is not _ForceBatch:
        return "force-aware samples must come from a validated force-data reader"
    if any(
        not any(issuer.contract is contract for contract in accepted_contracts)
        for issuer, _indices in batch._selections
    ):
        return "force-aware sample does not belong to the configured readers"
    offset = 0
    for issuer, indices in batch._selections:
        stop = offset + len(indices)
        for name, value in batch._values.items():
            expected = issuer.fingerprints.get(name)
            if expected is None or any(
                _tensor_fingerprint(value[offset + position]) != expected[index]
                for position, index in enumerate(indices)
            ):
                return "force-aware sample content does not match its reader frames"
        offset = stop
    return None


def _slice_selections(selections, start, stop):
    result = []
    offset = 0
    for issuer, indices in selections:
        selection_stop = offset + len(indices)
        lower = max(start, offset)
        upper = min(stop, selection_stop)
        if lower < upper:
            result.append((issuer, indices[lower - offset : upper - offset]))
        offset = selection_stop
    return tuple(result)


def concat_batch(tdicts, dim=0):
    keys = tdicts[0].keys()
    assert all(d.keys() == keys for d in tdicts)
    values = {k: torch.cat([d[k] for d in tdicts], dim) for k in keys}
    strict = [type(batch) is _ForceBatch for batch in tdicts]
    if any(strict) and not all(strict):
        raise ValueError("cannot concatenate reader-issued and ordinary batches")
    if strict and strict[0]:
        selections = tuple(
            selection
            for batch in tdicts
            for selection in batch._selections
        )
        return _ForceBatch._from_issued(values, selections)
    return values


def split_batch(tdict, size, dim=0):
    dsplit = {k: torch.split(v, size, dim) for k, v in tdict.items()}
    nsecs = [len(v) for v in dsplit.values()]
    assert all(ns == nsecs[0] for ns in nsecs)
    batches = [{k: v[i] for k, v in dsplit.items()} for i in range(nsecs[0])]
    if type(tdict) is not _ForceBatch:
        return batches
    result = []
    start = 0
    for batch in batches:
        stop = start + next(iter(batch.values())).shape[dim]
        result.append(
            _ForceBatch._from_issued(
                batch,
                _slice_selections(tdict._selections, start, stop),
            )
        )
        start = stop
    return result


class Reader(object):
    def __init__(self, data_path, batch_size,
                 energy_name="e_corr_target", descriptor_name="descriptor",
                 force_name=None, jacobian_name=None,
                 reference_orbital_gradient_name="reference_orbital_gradient",
                 descriptor_orbital_gradient_jacobian_name="descriptor_orbital_gradient_jacobian",
                 coulomb_loss_descriptor_gradient_name="coulomb_loss_descriptor_gradient",
                 converged_name="converged", atom_name="atom",
                 converged_filter=True, force_mode=FORCE_MODE_NONE, **kwargs):
        self.data_path = data_path
        self.batch_size = batch_size
        if force_mode not in FORCE_DATA_MODES:
            raise ValueError(
                f"force_mode must be one of {sorted(FORCE_DATA_MODES)}"
            )
        self.force_mode = force_mode
        self.converged_filter = converged_filter
        self.force_contract = None
        self._force_batch_issuer = None
        self._force_arrays = None
        strict_manifest_path = os.path.join(data_path, SCHEMA_FILENAME)
        if force_mode == FORCE_MODE_DEEPHF_RELAXED:
            if energy_name != "e_corr_target" or descriptor_name != "descriptor":
                raise ForceDataError(
                    "strict DeePHF force data uses canonical e_corr_target and descriptor fields"
                )
            if force_name not in (None, "f_corr_target"):
                raise ForceDataError(
                    "strict DeePHF force data uses canonical f_corr_target"
                )
            if jacobian_name not in (None, "dq_dR_relaxed"):
                raise ForceDataError(
                    "strict DeePHF force data uses canonical dq_dR_relaxed"
                )
            self.force_contract, self._force_arrays = load_force_dataset(data_path)
        elif os.path.isfile(strict_manifest_path):
            raise ForceDataError(
                "a strict force dataset must be read with "
                "force_mode='deephf_relaxed'"
            )
        elif force_name is not None or jacobian_name is not None:
            raise ForceDataError(
                "force field names require force_mode='deephf_relaxed'; "
                "fixed-density Jacobians are not force-training inputs"
            )
        self.energy_path = self.check_exist(energy_name + ".npy")
        self.descriptor_path = self.check_exist(descriptor_name + ".npy")
        self.force_path = None
        self.jacobian_path = None
        self.reference_orbital_gradient_path = self.check_exist(
            reference_orbital_gradient_name + ".npy"
        )
        self.descriptor_orbital_gradient_jacobian_path = self.check_exist(
            descriptor_orbital_gradient_jacobian_name + ".npy"
        )
        self.coulomb_loss_descriptor_gradient_path = self.check_exist(
            coulomb_loss_descriptor_gradient_name + ".npy"
        )
        self.converged_path = self.check_exist(converged_name + ".npy")
        self.atom_path = self.check_exist(atom_name + ".npy")
        # load data
        self.load_meta()
        self.prepare()
        if self.force_contract is not None:
            issuer = object.__new__(_ForceBatchIssuer)
            object.__setattr__(issuer, "contract", self.force_contract)
            object.__setattr__(
                issuer,
                "fingerprints",
                MappingProxyType(
                    {
                        name: tuple(
                            _tensor_fingerprint(value[index])
                            for index in range(self.nframes)
                        )
                        for name, value in self.tensor_data.items()
                    }
                ),
            )
            object.__setattr__(issuer, "frame_count", self.nframes)
            _FORCE_BATCH_ISSUERS.add(issuer)
            self._force_batch_issuer = issuer
        # initialize sample index queue
        self.idx_queue = []

    def check_exist(self, fname):
        if fname is None:
            return None
        fpath = os.path.join(self.data_path, fname)
        if os.path.exists(fpath):
            return fpath

    def load_meta(self):
        if self.force_contract is not None:
            dimensions = getattr(self.force_contract, "dimensions", None)
            if dimensions is None:
                dimensions = self.force_contract.manifest["dimensions"]
            self.natm = int(dimensions["n_descriptor_atom"])
            self.nraw = int(dimensions["n_raw_atom"])
            self.nproj = int(dimensions["n_feature"])
            self.descriptor_size = self.nproj
            return
        try:
            sys_meta = np.loadtxt(self.check_exist('system.raw'), dtype = int).reshape([-1])
            self.natm = sys_meta[0]
            self.nraw = sys_meta[1] if sys_meta.size > 1 else self.natm
            self.nproj = sys_meta[-1]
        except:
            print('#', self.data_path, f"no system.raw, infer meta from data", file=sys.stderr)
            sys_shape = np.load(self.descriptor_path).shape
            assert len(sys_shape) == 3, \
                f"descriptor has to be an order-3 array with shape [nframes, natom, nproj]"
            self.natm = sys_shape[1]
            self.nraw = self.natm
            self.nproj = sys_shape[2]
        self.descriptor_size = self.nproj

    def prepare(self):
        if self.force_contract is not None:
            self._prepare_force_data()
            return
        # load energy and check nframes
        energy = np.load(self.energy_path).reshape(-1, 1)
        raw_nframes = energy.shape[0]
        descriptor = np.load(self.descriptor_path).reshape(
            raw_nframes, self.natm, self.descriptor_size
        )
        if self.converged_filter and self.converged_path is not None:
            converged = np.load(self.converged_path).reshape(raw_nframes)
        else:
            converged = np.ones(raw_nframes, dtype=bool)
        self.data_energy = energy[converged]
        self.data_descriptor = descriptor[converged]
        self.nframes = converged.sum()
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print('#', self.data_path, 
                 f"reset batch size to {self.batch_size}", file=sys.stderr)
        # handle atom and element data
        self.atom_info = {}
        if self.atom_path is not None:
            atoms = np.load(self.atom_path).reshape(raw_nframes, self.natm, 4)
            self.atom_info["elements"] = atoms[:, :, 0][converged].round().astype(int)
            self.atom_info["coordinates"] = atoms[:, :, 1:][converged]
        # load data in torch
        self.tensor_data = {
            "energy": torch.from_numpy(self.data_energy),
            "descriptor": torch.from_numpy(self.data_descriptor),
        }
        if (
            self.reference_orbital_gradient_path is not None
            and self.descriptor_orbital_gradient_jacobian_path is not None
        ):
            reference_gradient = np.load(self.reference_orbital_gradient_path).reshape(
                raw_nframes, -1
            )[converged]
            descriptor_jacobian = np.load(
                self.descriptor_orbital_gradient_jacobian_path
            ).reshape(raw_nframes, self.natm, self.descriptor_size, -1)[converged]
            self.tensor_data["reference_orbital_gradient"] = torch.from_numpy(
                reference_gradient
            )
            self.tensor_data["descriptor_orbital_gradient_jacobian"] = torch.from_numpy(
                descriptor_jacobian
            )
            self.orbital_gradient_size = self.tensor_data[
                "reference_orbital_gradient"
            ].shape[-1]
        if self.coulomb_loss_descriptor_gradient_path is not None:
            coulomb_gradient = np.load(
                self.coulomb_loss_descriptor_gradient_path
            ).reshape(raw_nframes, self.natm, self.descriptor_size)[converged]
            self.tensor_data["coulomb_loss_descriptor_gradient"] = torch.from_numpy(
                coulomb_gradient
            )

    def _prepare_force_data(self):
        arrays = self._force_arrays
        energy = arrays["e_corr_target"]
        descriptor = arrays["descriptor"]
        force = arrays["f_corr_target"]
        jacobian = arrays["dq_dR_relaxed"]
        atoms = arrays["atom"]
        self.data_energy = energy
        self.data_descriptor = descriptor
        self.nframes = int(energy.shape[0])
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print(
                '#', self.data_path,
                f"reset batch size to {self.batch_size}",
                file=sys.stderr,
            )
        self.atom_info = {
            "elements": atoms[:, :, 0].round().astype(int),
            "coordinates": atoms[:, :, 1:],
        }
        self.tensor_data = {
            "energy": torch.from_numpy(energy),
            "descriptor": torch.from_numpy(descriptor),
            "force": torch.from_numpy(force),
            "dq_dR_relaxed": torch.from_numpy(jacobian),
        }
        self._force_arrays = None

    def sample_train(self):
        if self.batch_size == self.nframes == 1:
            return self.sample_all()
        if len(self.idx_queue) < self.batch_size:
            self.idx_queue = np.random.choice(self.nframes, self.nframes, replace=False)
        sample_idx = self.idx_queue[:self.batch_size]
        self.idx_queue = self.idx_queue[self.batch_size:]
        values = {k: v[sample_idx] for k, v in self.tensor_data.items()}
        if self.force_contract is not None:
            return _ForceBatch._from_issued(
                values,
                ((self._force_batch_issuer, tuple(map(int, sample_idx))),),
            )
        return values

    def sample_all(self):
        if self.force_contract is not None:
            return _ForceBatch._from_issued(
                self.tensor_data,
                ((self._force_batch_issuer, range(self.nframes)),),
            )
        return self.tensor_data

    def get_train_size(self):
        return self.nframes

    def get_batch_size(self):
        return self.batch_size

    def get_nframes(self):
        return self.nframes
    
    def collect_elems(self, elem_list):
        if "elem_list" in self.atom_info:
            assert list(elem_list) == list(self.atom_info["elem_list"])
            return self.atom_info["nelem"]
        elem_to_idx = np.zeros(200, dtype=int) + 200
        for ii, ee in enumerate(elem_list):
            elem_to_idx[ee] = ii
        idxs = elem_to_idx[self.atom_info["elements"]]
        nelem = np.zeros((self.nframes, len(elem_list)), int)
        np.add.at(nelem, (np.arange(nelem.shape[0]).reshape(-1,1), idxs), 1)
        self.atom_info["nelem"] = nelem
        self.atom_info["elem_list"] = elem_list
        return nelem
    
    def subtract_elem_const(self, elem_const):
        # assert "elem_const" not in self.atom_info, \
        #     "subtract_elem_const has been done. The method should not be executed twice."
        econst = (self.atom_info["nelem"] @ elem_const).reshape(self.nframes, 1)
        self.data_energy -= econst
        self.atom_info["elem_const"] = elem_const
    
    def revert_elem_const(self):
        # assert "elem_const" not in self.atom_info, \
        #     "subtract_elem_const has been done. The method should not be executed twice."
        if "elem_const" not in self.atom_info:
            return
        elem_const = self.atom_info.pop("elem_const")
        econst = (self.atom_info["nelem"] @ elem_const).reshape(self.nframes, 1)
        self.data_energy += econst
        

class GroupReader(object) :
    def __init__ (self, path_list, batch_size=1, group_batch=1, extra_label=True, **kwargs):
        if isinstance(path_list, str):
            path_list = [path_list]
        self.path_list = path_list
        self.batch_size = batch_size
        # init system readers
        force_mode = kwargs.get("force_mode", FORCE_MODE_NONE)
        if force_mode not in FORCE_DATA_MODES:
            raise ValueError(
                f"force_mode must be one of {sorted(FORCE_DATA_MODES)}"
            )
        strict_paths = [
            path
            for path in self.path_list
            if os.path.isfile(os.path.join(path, SCHEMA_FILENAME))
        ]
        if strict_paths and force_mode != FORCE_MODE_DEEPHF_RELAXED:
            raise ForceDataError(
                "strict force datasets must be grouped with "
                "force_mode='deephf_relaxed'"
            )
        if force_mode == FORCE_MODE_DEEPHF_RELAXED and (
            not extra_label
            or not isinstance(kwargs.get('descriptor_name', "descriptor"), str)
        ):
            raise ForceDataError(
                "strict DeePHF force data requires the canonical Reader path"
            )
        Reader_class = (Reader if extra_label
            and isinstance(kwargs.get('descriptor_name', "descriptor"), str)
            else SimpleReader)
        self.readers = []
        self.nframes = []
        for ipath in self.path_list :
            ireader = Reader_class(ipath, batch_size, **kwargs)
            if ireader.get_nframes() == 0:
                print('# ignore empty dataset:', ipath, file=sys.stderr)
                continue
            self.readers.append(ireader)
            self.nframes.append(ireader.get_nframes())
        if not self.readers:
            raise RuntimeError("No system is avaliable")
        self.nsystems = len(self.readers)
        data_keys = self.readers[0].sample_all().keys()
        if any(reader.sample_all().keys() != data_keys for reader in self.readers[1:]):
            raise ValueError("all grouped datasets must expose the same fields")
        print(f"# load {self.nsystems} systems with fields {set(data_keys)}")
        # probability of each system
        self.descriptor_size = self.readers[0].descriptor_size
        if any(
            reader.descriptor_size != self.descriptor_size
            for reader in self.readers[1:]
        ):
            raise ValueError("all grouped datasets must use the same descriptor size")
        self.force_contract = getattr(self.readers[0], "force_contract", None)
        self.force_contracts = tuple(
            reader.force_contract
            for reader in self.readers
            if reader.force_contract is not None
        )
        if force_mode == FORCE_MODE_DEEPHF_RELAXED:
            fingerprint = getattr(self.force_contract, "fingerprint", None)
            if fingerprint is None:
                fingerprint = self.force_contract.compatibility_fingerprint
            for reader in self.readers[1:]:
                other_fingerprint = getattr(reader.force_contract, "fingerprint", None)
                if other_fingerprint is None:
                    other_fingerprint = reader.force_contract.compatibility_fingerprint
                if other_fingerprint != fingerprint:
                    raise ForceDataError(
                        "grouped force datasets have incompatible provenance contracts"
                    )
        self.sys_prob = [float(ii) for ii in self.nframes] / np.sum(self.nframes)
        
        self.group_batch = max(group_batch, 1)
        if self.group_batch > 1:
            self.group_dict = {}
            # self.group_index = {}
            for idx, r in enumerate(self.readers):
                shape = (
                    getattr(r, "nraw", r.natm),
                    r.natm,
                    getattr(r, "orbital_gradient_size", None),
                )
                if shape in self.group_dict:
                    self.group_dict[shape].append(r)
                    # self.group_index[shape].append(idx)
                else:
                    self.group_dict[shape] = [r]
                    # self.group_index[shape] = [idx]
            self.group_prob = {n: sum(r.nframes for r in r_list) / sum(self.nframes)
                                for n, r_list in self.group_dict.items()}
            self.batch_prob_raw = {n: [r.nframes / r.batch_size for r in r_list] 
                                for n, r_list in self.group_dict.items()}
            self.batch_prob = {n: p / np.sum(p) for n, p in self.batch_prob_raw.items()}

        self._sample_used = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._sample_used >= self.get_train_size():
            self._sample_used = 0
            raise StopIteration
        sample = self.sample_train() if self.group_batch == 1 else self.sample_train_group()
        self._sample_used += sample["energy"].shape[0]
        return sample

    def sample_idx(self) :
        return np.random.choice(np.arange(self.nsystems), p=self.sys_prob)
        
    def sample_train(self, idx=None) :
        if idx is None:
            idx = self.sample_idx()
        return \
            self.readers[idx].sample_train()

    def sample_train_group(self):
        cidx = np.random.choice(len(self.group_prob), p=list(self.group_prob.values()))
        cshape = list(self.group_prob.keys())[cidx]
        cgrp = self.group_dict[cshape]
        csys = np.random.choice(cgrp, self.group_batch, p=self.batch_prob[cshape])
        batch = concat_batch([s.sample_train() for s in csys], dim=0)
        return batch

    def sample_all(self, idx=None) :
        if idx is None:
            idx = self.sample_idx()
        return \
            self.readers[idx].sample_all()
    
    def sample_all_batch(self, idx=None):
        if idx is not None:
            all_data = self.sample_all(idx)
            size = self.batch_size * self.group_batch
            yield from split_batch(all_data, size, dim=0)
        else:
            for i in range(self.nsystems):
                yield from self.sample_all_batch(i)

    def get_train_size(self) :
        return np.sum(self.nframes)

    def get_batch_size(self) :
        return self.batch_size

    def compute_data_stat(self, symm_sections=None):
        all_descriptors = np.concatenate([
            reader.data_descriptor.reshape(-1, reader.descriptor_size)
            for reader in self.readers
        ])
        if symm_sections is None:
            all_mean, all_std = all_descriptors.mean(0), all_descriptors.std(0)
        else:
            assert sum(symm_sections) == all_descriptors.shape[-1]
            descriptor_shells = np.split(
                all_descriptors, np.cumsum(symm_sections)[:-1], axis=-1
            )
            mean_shells = [
                descriptor.mean().repeat(size)
                for descriptor, size in zip(descriptor_shells, symm_sections)
            ]
            std_shells = [
                descriptor.std().repeat(size)
                for descriptor, size in zip(descriptor_shells, symm_sections)
            ]
            all_mean = np.concatenate(mean_shells, axis=-1)
            all_std = np.concatenate(std_shells, axis=-1)
        return all_mean, all_std

    def compute_prefitting(self, shift=None, scale=None, ridge_alpha=1e-8, symm_sections=None):
        if shift is None or scale is None:
            all_mean, all_std = self.compute_data_stat(symm_sections=symm_sections)
            if shift is None:
                shift = all_mean
            if scale is None:
                scale = all_std
        all_scaled_descriptors = np.concatenate([
            ((reader.data_descriptor - shift) / scale).sum(1)
            for reader in self.readers
        ])
        all_natm = np.concatenate([
            [float(reader.data_descriptor.shape[1])] * reader.data_descriptor.shape[0]
            for reader in self.readers
        ])
        if symm_sections is not None: # in this case ridge alpha cannot be 0
            assert sum(symm_sections) == all_scaled_descriptors.shape[-1]
            descriptor_shells = np.split(
                all_scaled_descriptors, np.cumsum(symm_sections)[:-1], axis=-1
            )
            all_scaled_descriptors = np.stack(
                [descriptor.sum(-1) for descriptor in descriptor_shells], axis=-1
            )
        # build feature matrix
        X = np.concatenate([all_scaled_descriptors, all_natm.reshape(-1,1)], -1)
        y = np.concatenate([reader.data_energy for reader in self.readers])
        I = np.identity(X.shape[1])
        I[-1,-1] = 0 # do not punish the bias term
        # solve ridge reg
        coef = np.linalg.solve(X.T @ X + ridge_alpha * I, X.T @ y).reshape(-1)
        weight, bias = coef[:-1], coef[-1]
        if symm_sections is not None:
            weight = np.concatenate([w.repeat(s) for w, s in zip(weight, symm_sections)], axis=-1)
        return weight, bias
    
    def collect_elems(self, elem_list=None):
        if elem_list is None:
            elem_list = np.array(sorted(set.union(*[
                set(r.atom_info["elements"].flatten()) for r in self.readers
            ])))
        for rd in self.readers:
            rd.collect_elems(elem_list)
        return elem_list

    def compute_elem_const(self, ridge_alpha=0.):
        elem_list = self.collect_elems()
        all_nelem = np.concatenate([r.atom_info["nelem"] for r in self.readers])
        all_energy = np.concatenate([reader.data_energy for reader in self.readers])
        # lex sort by nelem
        lexidx = np.lexsort(all_nelem.T)
        all_nelem = all_nelem[lexidx]
        all_energy = all_energy[lexidx]
        # group by nelem
        _, sidx = np.unique(all_nelem, return_index=True, axis=0)
        sidx = np.sort(sidx)
        grp_nelem = all_nelem[sidx]
        group_energy = np.array(
            list(map(np.mean, np.split(all_energy, sidx[1:])))
        )
        if ridge_alpha <= 0:
            elem_const, _res, _rank, _sing = np.linalg.lstsq(
                grp_nelem, group_energy, None
            )
        else:
            I = np.identity(grp_nelem.shape[1])
            elem_const = np.linalg.solve(
                grp_nelem.T @ grp_nelem + ridge_alpha * I,
                grp_nelem.T @ group_energy,
            )
        return elem_list.reshape(-1), elem_const.reshape(-1)
    
    def subtract_elem_const(self, elem_const):
        for rd in self.readers:
            rd.subtract_elem_const(elem_const)
    
    def revert_elem_const(self):
        for rd in self.readers:
            rd.revert_elem_const()


class SimpleReader(object):
    def __init__(self, data_path, batch_size,
                 energy_name="e_corr_target", descriptor_name="descriptor",
                 converged_filter=True, converged_name="converged",
                 force_mode=FORCE_MODE_NONE, **kwargs):
        # copy from config
        self.data_path = data_path
        self.batch_size = batch_size
        if force_mode not in FORCE_DATA_MODES:
            raise ValueError(
                f"force_mode must be one of {sorted(FORCE_DATA_MODES)}"
            )
        if os.path.isfile(os.path.join(data_path, SCHEMA_FILENAME)):
            if force_mode != FORCE_MODE_DEEPHF_RELAXED:
                raise ForceDataError(
                    "a strict force dataset must be read with "
                    "force_mode='deephf_relaxed'"
                )
            raise ForceDataError(
                "strict DeePHF force data requires the canonical Reader path"
            )
        if force_mode == FORCE_MODE_DEEPHF_RELAXED:
            raise ForceDataError(
                "strict DeePHF force data requires the canonical Reader path"
            )
        self.energy_name = energy_name
        self.descriptor_names = (
            descriptor_name
            if isinstance(descriptor_name, (list, tuple))
            else [descriptor_name]
        )
        self.converged_filter = converged_filter
        self.converged_name = converged_name
        self.load_meta()
        self.prepare()

    def load_meta(self):
        try:
            sys_meta = np.loadtxt(os.path.join(self.data_path,'system.raw'), dtype = int).reshape([-1])
            self.natm = sys_meta[0]
            self.nproj = sys_meta[-1]
        except:
            print('#', self.data_path, f"no system.raw, infer meta from data", file=sys.stderr)
            sys_shape = np.load(
                os.path.join(self.data_path, f'{self.descriptor_names[0]}.npy')
            ).shape
            assert len(sys_shape) == 3, \
                f"{self.descriptor_names[0]} has to be an order-3 array with shape [nframes, natom, nproj]"
            self.natm = sys_shape[1]
            self.nproj = sys_shape[2]
    
    def prepare(self):
        self.sample_index_end = 0
        energy = np.load(
            os.path.join(self.data_path, f'{self.energy_name}.npy')
        ).reshape([-1, 1])
        raw_nframes = energy.shape[0]
        descriptor = np.concatenate(
            [np.load(os.path.join(self.data_path, f'{name}.npy'))
               .reshape([raw_nframes, self.natm, -1])
            for name in self.descriptor_names],
            axis=-1)
        if self.converged_filter:
            converged = np.load(
                os.path.join(self.data_path, f'{self.converged_name}.npy')
            ).reshape(raw_nframes)
        else:
            converged = np.ones(raw_nframes, dtype=bool)
        self.data_energy = energy[converged]
        self.data_descriptor = descriptor[converged]
        self.nframes = converged.sum()
        self.descriptor_size = self.data_descriptor.shape[-1]
        # print(np.shape(self.inputs_train))
        if self.nframes < self.batch_size:
            self.batch_size = self.nframes
            print('#', self.data_path, f"reset batch size to {self.batch_size}", file=sys.stderr)
    
    def sample_train(self):
        if self.nframes == self.batch_size == 1:
            return self.sample_all()
        self.sample_index_end += self.batch_size
        if self.sample_index_end > self.nframes:
            # shuffle the data
            self.sample_index_end = self.batch_size
            ind = np.random.choice(self.nframes, self.nframes, replace=False)
            self.data_energy = self.data_energy[ind]
            self.data_descriptor = self.data_descriptor[ind]
        ind = np.arange(self.sample_index_end - self.batch_size, self.sample_index_end)
        return {
            "energy": torch.from_numpy(self.data_energy[ind]),
            "descriptor": torch.from_numpy(self.data_descriptor[ind])
        }

    def sample_all(self) :
        return {
            "energy": torch.from_numpy(self.data_energy),
            "descriptor": torch.from_numpy(self.data_descriptor)
        }

    def get_train_size(self) :
        return self.nframes

    def get_batch_size(self) :
        return self.batch_size

    def get_nframes(self) :
        return self.nframes
