"""Public construction and inference workflow for strict DeePHF methods."""

from collections.abc import Mapping
import json
import os
from pathlib import Path

import numpy as np
from pyscf import dft, scf

from deepks.data.io import build_molecule, dump_data, iter_system
from deepks.gpu import (
    DEFAULT_CUDA_DEVICE,
    GPU_DIRECT_SCF_TOL,
    as_numpy,
    require_cuda_device,
)
from deepks.model.model import CorrNet
from deepks.utils import get_sys_name, load_sys_paths

from .method import DeePHF
from .contracts import (
    RootContinuityError,
    occupied_coefficients,
    occupied_subspace_overlaps,
    validate_root_overlap_tolerance,
)
from .pyscf_rhf_reference import _molecule_static_fingerprint, reference_fingerprint
from .pyscf_rks_reference import rks_reference_fingerprint
from .rks_method import RKSDeePHF
from .unrestricted_method import UHFDeePHF, UKSDeePHF
from .unrestricted_reference import (
    uhf_reference_fingerprint,
    uks_reference_fingerprint,
)
from .gpu_method import (
    GPU_METHOD_CLASSES,
    gpu_reference_family,
    is_gpu_reference,
)


METHOD_CLASSES = {
    scf.hf.RHF: DeePHF,
    scf.uhf.UHF: UHFDeePHF,
    dft.rks.RKS: RKSDeePHF,
    dft.uks.UKS: UKSDeePHF,
}
REFERENCE_FAMILIES = frozenset(("rhf", "uhf", "rks", "uks"))
INFERENCE_PROVENANCE_FILENAME = "deephf_provenance.json"


def _canonicalize_final_orbitals(reference) -> None:
    """Canonicalize the converged final Fock matrix without changing its density."""
    density = reference.make_rdm1(reference.mo_coeff, reference.mo_occ)
    fock = reference.get_fock(dm=density)
    reference.mo_energy, reference.mo_coeff = reference.canonicalize(
        reference.mo_coeff,
        reference.mo_occ,
        fock=fock,
    )


def _normalize_gpu_dft_args(dft_args) -> dict:
    """Return the serializable DFT controls used by every trajectory frame."""
    if dft_args is None:
        return {
            "xc": "LDA_X + LDA_C_VWN",
            "grid_mode": "strict",
            "grid_level": 3,
            "small_rho_cutoff": 0.0,
        }
    if not isinstance(dft_args, Mapping):
        raise TypeError("dft_args must be a mapping")
    controls = {
        "xc": "LDA_X + LDA_C_VWN",
        "grid_mode": "default",
        "grid_level": 3,
        "small_rho_cutoff": 0.0,
        **dict(dft_args),
    }
    supported = {"xc", "grid_mode", "grid_level", "small_rho_cutoff"}
    unknown = sorted(set(controls) - supported)
    if unknown:
        raise ValueError("unsupported GPU DFT controls: " + ", ".join(unknown))
    if not isinstance(controls["xc"], str) or not controls["xc"].strip():
        raise ValueError("GPU DFT xc must be a nonempty string")
    if controls["grid_mode"] not in {"default", "strict"}:
        raise ValueError("GPU DFT grid_mode must be 'default' or 'strict'")
    if type(controls["grid_level"]) is not int or controls["grid_level"] < 0:
        raise ValueError("GPU DFT grid_level must be a nonnegative integer")
    small_rho_cutoff = float(controls["small_rho_cutoff"])
    if small_rho_cutoff < 0.0 or not np.isfinite(small_rho_cutoff):
        raise ValueError("GPU DFT small_rho_cutoff must be finite and nonnegative")
    controls["small_rho_cutoff"] = small_rho_cutoff
    return controls


def _configure_gpu_dft(reference, molecule, dft_args) -> dict:
    """Configure one reproducible GPU DFT functional and integration grid."""
    controls = _normalize_gpu_dft_args(dft_args)
    reference.xc = controls["xc"]
    reference.grids.level = controls["grid_level"]
    reference.grids.cutoff = 1.0e-15
    if controls["grid_mode"] == "strict":
        reference.grids.alignment = 1
        reference.grids.atom_grid = {
            symbol: (20, 50) for symbol in set(molecule.elements)
        }
        reference.grids.prune = None
        reference.grids.build(with_non0tab=True, sort_grids=False)
    else:
        reference.grids.build(with_non0tab=True)
    reference.small_rho_cutoff = controls["small_rho_cutoff"]
    return controls


def make_deephf(
    reference,
    model,
    *,
    projector_basis=None,
    device=DEFAULT_CUDA_DEVICE,
    response_options=None,
    adjoint_options=None,
):
    """Construct the exact public DeePHF method matching a native reference."""
    family = gpu_reference_family(reference)
    method_class = (
        GPU_METHOD_CLASSES.get(family)
        if family is not None
        else METHOD_CLASSES.get(type(reference))
    )
    if method_class is None:
        raise TypeError("the reference type has no strict DeePHF method")
    if projector_basis is None and model is not None:
        projector_basis = model._pbas
    return method_class(
        reference,
        model,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )


def build_reference(
    molecule,
    family,
    *,
    scf_args=None,
    dft_args=None,
    dm0=None,
    verbose=0,
):
    """Converge one GPU4PySCF reference and return its strict PySCF state."""
    require_cuda_device()
    import cupy
    from gpu4pyscf import dft as gpu_dft
    from gpu4pyscf import scf as gpu_scf

    if type(family) is not str or family.lower() not in REFERENCE_FAMILIES:
        raise ValueError("reference family must be one of rhf, uhf, rks, or uks")
    family = family.lower()
    if family not in {"rks", "uks"} and dft_args is not None:
        raise ValueError("dft_args require an RKS or UKS reference family")
    if family in {"rhf", "rks"} and molecule.spin != 0:
        raise ValueError(f"the strict {family.upper()} family requires molecular spin zero")
    if family == "rhf":
        reference = gpu_scf.RHF(molecule)
    elif family == "uhf":
        reference = gpu_scf.UHF(molecule)
    elif family == "rks":
        reference = gpu_dft.RKS(molecule)
    else:
        reference = gpu_dft.UKS(molecule)
    if family in {"uhf", "uks"}:
        reference.init_guess_breaksym = 1
    reference.verbose = verbose
    controls = {
        "conv_tol": 1.0e-12,
        "conv_tol_grad": 1.0e-10,
        "conv_tol_cpscf": 1.0e-12,
        "direct_scf_tol": GPU_DIRECT_SCF_TOL,
        "max_cycle": 100,
        "newton_max_cycle": 50,
    }
    controls.update({} if scf_args is None else dict(scf_args))
    newton_max_cycle = controls.pop("newton_max_cycle")
    if type(newton_max_cycle) is not int or newton_max_cycle < 0:
        raise ValueError("newton_max_cycle must be a nonnegative integer")
    supported_controls = {
        "conv_tol",
        "conv_tol_grad",
        "conv_tol_cpscf",
        "max_cycle",
        "diis_space",
        "diis_start_cycle",
        "damp",
        "init_guess",
        "init_guess_breaksym",
        "level_shift",
        "direct_scf",
        "direct_scf_tol",
        "conv_check",
    }
    unknown = sorted(set(controls) - supported_controls)
    if unknown:
        raise ValueError("unsupported strict reference controls: " + ", ".join(unknown))
    reference.set(**controls)
    active_dft_args = None
    if family in {"rks", "uks"}:
        active_dft_args = _configure_gpu_dft(reference, molecule, dft_args)
    if dm0 is not None:
        initial_density = cupy.asarray(dm0)
        expected_shape = (
            (molecule.nao_nr(), molecule.nao_nr())
            if family in {"rhf", "rks"}
            else (2, molecule.nao_nr(), molecule.nao_nr())
        )
        if (
            initial_density.shape != expected_shape
            or cupy.iscomplexobj(initial_density)
            or not bool(cupy.isfinite(initial_density).all().item())
        ):
            raise ValueError(
                f"the {family.upper()} initial density must be real, finite, "
                f"and have shape {expected_shape}"
            )
        initial_density = cupy.ascontiguousarray(
            initial_density, dtype=cupy.float64
        )
    else:
        initial_density = None
    reference.kernel(dm0=initial_density)
    if not reference.converged and newton_max_cycle:
        second_order = reference.newton()
        second_order.max_cycle = newton_max_cycle
        second_order.kernel()
        if second_order.converged:
            reference.converged = True
            reference.e_tot = second_order.e_tot
            reference.mo_energy = second_order.mo_energy
            reference.mo_coeff = second_order.mo_coeff
            reference.mo_occ = second_order.mo_occ
    if not reference.converged:
        raise RuntimeError(f"the GPU4PySCF {family.upper()} reference did not converge")
    _canonicalize_final_orbitals(reference)
    if gpu_reference_family(reference) != family:
        raise TypeError(
            f"the converged reference is not an exact GPU4PySCF {family.upper()} object"
        )
    reference._deepks_scf_args = {
        **controls,
        "newton_max_cycle": newton_max_cycle,
    }
    reference._deepks_dft_args = active_dft_args
    return reference


def evaluate_molecule(
    molecule,
    model,
    *,
    family,
    backend="direct",
    projector_basis=None,
    device=DEFAULT_CUDA_DEVICE,
    scf_args=None,
    dft_args=None,
    response_options=None,
    adjoint_options=None,
    verbose=0,
):
    """Evaluate energy, descriptor, gradient, and force through one public backend."""
    reference = build_reference(
        molecule,
        family,
        scf_args=scf_args,
        dft_args=dft_args,
        verbose=verbose,
    )
    return _evaluate_reference(
        reference,
        model,
        backend=backend,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )


def _evaluate_reference(
    reference,
    model,
    *,
    backend="direct",
    projector_basis=None,
    device=DEFAULT_CUDA_DEVICE,
    response_options=None,
    adjoint_options=None,
):
    """Evaluate one already converged native reference without rebuilding it."""
    method = make_deephf(
        reference,
        model,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )
    with method._controlled_calculation():
        energy = float(method.kernel())
        gradient_driver = method.nuc_grad_method(
            backend=backend,
            retain_details=False,
        )
        gradient = np.asarray(gradient_driver.kernel())
        descriptor = np.asarray(method.descriptor(), dtype=np.float64)
        result = {
            "converged": np.asarray(True),
            "e_base": np.asarray(method.e_base, dtype=np.float64),
            "e_corr": np.asarray(method.e_corr, dtype=np.float64),
            "e_tot": np.asarray(energy, dtype=np.float64),
            "descriptor": descriptor,
            "gradient": gradient,
            "force": -gradient,
        }
    return result


def _reference_state_fingerprint(reference) -> str:
    if is_gpu_reference(reference):
        from .contracts import update_digest
        import hashlib

        digest = hashlib.sha256()
        update_digest(digest, gpu_reference_family(reference))
        update_digest(digest, getattr(reference, "xc", None))
        update_digest(digest, getattr(reference, "_deepks_dft_args", None))
        update_digest(digest, float(reference.e_tot))
        update_digest(digest, reference.mol.atom_coords(unit="Bohr"))
        update_digest(digest, as_numpy(reference.mo_energy))
        update_digest(digest, as_numpy(reference.mo_coeff))
        update_digest(digest, as_numpy(reference.mo_occ))
        return digest.hexdigest()
    fingerprint = {
        scf.hf.RHF: reference_fingerprint,
        scf.uhf.UHF: uhf_reference_fingerprint,
        dft.rks.RKS: rks_reference_fingerprint,
        dft.uks.UKS: uks_reference_fingerprint,
    }.get(type(reference))
    if fingerprint is None:
        raise TypeError("the reference type has no strict state fingerprint")
    return fingerprint(reference)


class _ReferenceSequence:
    """Track one accepted electronic root for a CLI molecular sequence."""

    def __init__(
        self,
        family,
        *,
        scf_args=None,
        dft_args=None,
        root_overlap_tolerance=0.5,
        verbose=0,
    ):
        if type(family) is not str or family.lower() not in REFERENCE_FAMILIES:
            raise ValueError("reference family must be one of rhf, uhf, rks, or uks")
        self.family = family.lower()
        self.scf_args = None if scf_args is None else dict(scf_args)
        self.dft_args = None if dft_args is None else dict(dft_args)
        self.root_overlap_tolerance = validate_root_overlap_tolerance(
            root_overlap_tolerance,
            owner="trajectory",
        )
        self.verbose = verbose
        self._previous_reference = None
        self._system_fingerprint = None
        self._previous_density = None
        self._previous_occupations = None
        self._previous_occupied = None
        self._previous_fingerprint = None
        self.records = []

    def build(self, molecule):
        """Build one candidate and advance the root anchor only after acceptance."""
        system_fingerprint = _molecule_static_fingerprint(molecule)
        if (
            self._system_fingerprint is not None
            and system_fingerprint != self._system_fingerprint
        ):
            raise RootContinuityError(
                "the molecular system or AO basis changed within one trajectory"
            )
        reference = build_reference(
            molecule,
            self.family,
            scf_args=self.scf_args,
            dft_args=self.dft_args,
            dm0=self._previous_density,
            verbose=self.verbose,
        )
        occupations = as_numpy(reference.mo_occ)
        candidate_occupied = occupied_coefficients(
            as_numpy(reference.mo_coeff),
            occupations,
        )
        state_fingerprint = _reference_state_fingerprint(reference)
        channel_names = (
            ("restricted",)
            if self.family in {"rhf", "rks"}
            else ("alpha", "beta")
        )
        if self._previous_reference is None:
            overlaps = tuple(1.0 for _ in channel_names)
            initial_guess_source = "independent"
        else:
            if not np.array_equal(occupations, self._previous_occupations):
                raise RootContinuityError(
                    f"the {self.family.upper()} occupations changed from the accepted root"
                )
            overlaps = occupied_subspace_overlaps(
                self._previous_reference.mol,
                self._previous_occupied,
                reference.mol,
                candidate_occupied,
            )
            initial_guess_source = "previous_density"
        minimum_overlap = float(min(overlaps))
        if minimum_overlap < self.root_overlap_tolerance:
            raise RootContinuityError(
                f"the {self.family.upper()} occupied subspace is discontinuous: "
                f"minimum overlap {minimum_overlap:.6f} < "
                f"{self.root_overlap_tolerance:.6f}"
            )
        record = {
            "frame_index": len(self.records),
            "reference_state_fingerprint": state_fingerprint,
            "parent_reference_state_fingerprint": self._previous_fingerprint,
            "initial_guess_source": initial_guess_source,
            "occupied_subspace_overlaps": {
                name: float(overlap)
                for name, overlap in zip(channel_names, overlaps)
            },
            "minimum_occupied_overlap": minimum_overlap,
        }
        density = reference.make_rdm1(reference.mo_coeff, reference.mo_occ)
        density_array = as_numpy(density)
        if np.iscomplexobj(density_array) or not np.isfinite(density_array).all():
            raise RootContinuityError(
                f"the accepted {self.family.upper()} density is invalid"
            )
        self._previous_reference = reference
        self._system_fingerprint = system_fingerprint
        self._previous_density = density.copy()
        self._previous_occupations = np.ascontiguousarray(occupations).copy()
        self._previous_occupied = tuple(value.copy() for value in candidate_occupied)
        self._previous_fingerprint = state_fingerprint
        self.records.append(record)
        return reference


def _model_target_provenance(model):
    if model is None:
        return None
    extra_info = getattr(model, "_checkpoint_extra_info", None)
    if not isinstance(extra_info, Mapping):
        return None
    metadata = extra_info.get("force_training")
    if not isinstance(metadata, Mapping):
        return None
    from deepks.data.force_schema import (
        normalize_target_identity,
        target_identity_fingerprint,
    )

    target = normalize_target_identity(metadata["target"])
    fingerprint = target_identity_fingerprint(target)
    if metadata.get("target_fingerprint") != fingerprint:
        raise ValueError(
            "force-training checkpoint target fingerprint changed after loading"
        )
    return {"identity": target, "fingerprint": fingerprint}


def _write_inference_provenance(directory, payload) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    destination = path / INFERENCE_PROVENANCE_FILENAME
    temporary = path / f".{INFERENCE_PROVENANCE_FILENAME}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main(
    systems,
    *,
    reference,
    model_file=None,
    basis="sto-3g",
    projector_basis=None,
    backend="direct",
    device=DEFAULT_CUDA_DEVICE,
    dump_dir=".",
    mol_args=None,
    scf_args=None,
    dft_args=None,
    response_options=None,
    adjoint_options=None,
    root_overlap_tolerance=0.5,
    verbose=0,
):
    """Run strict DeePHF inference for every frame and persist canonical outputs."""
    if model_file is None or str(model_file).upper() == "NONE":
        model = None
    else:
        model = CorrNet.load(model_file, strict=True).double().eval()
    mol_args = {} if mol_args is None else dict(mol_args)
    outputs = []
    for system_path in load_sys_paths(systems):
        frames = []
        sequence = _ReferenceSequence(
            reference,
            scf_args=scf_args,
            dft_args=dft_args,
            root_overlap_tolerance=root_overlap_tolerance,
            verbose=verbose,
        )
        for atoms, attributes, _labels in iter_system(system_path):
            molecule_input = {
                **mol_args,
                "atom": atoms,
                "basis": basis,
                "verbose": verbose,
                **attributes,
            }
            molecule = build_molecule(**molecule_input)
            native_reference = sequence.build(molecule)
            frames.append(
                _evaluate_reference(
                    native_reference,
                    model,
                    backend=backend,
                    projector_basis=projector_basis,
                    device=device,
                    response_options=response_options,
                    adjoint_options=adjoint_options,
                )
            )
        if not frames:
            raise ValueError(f"system {system_path!r} contains no molecular frames")
        collected = {name: np.asarray([frame[name] for frame in frames]) for name in frames[0]}
        output_directory = os.path.join(dump_dir, get_sys_name(system_path))
        provenance = {
            "schema": {
                "id": "deepks.deephf.inference-provenance",
                "version": 1,
            },
            "reference_family": sequence.family.upper(),
            "root_overlap_tolerance": sequence.root_overlap_tolerance,
            "frames": sequence.records,
        }
        model_target = _model_target_provenance(model)
        if model_target is not None:
            provenance["model_target"] = model_target
        dump_data(output_directory, **collected)
        _write_inference_provenance(output_directory, provenance)
        outputs.append((output_directory, collected))
    return outputs


__all__ = [
    "INFERENCE_PROVENANCE_FILENAME",
    "build_reference",
    "evaluate_molecule",
    "main",
    "make_deephf",
]
