"""Strict capability checks for perturbative DeePHF references and models."""

from collections.abc import Mapping
import hashlib
import random

import numpy as np
import torch
from pyscf import gto, scf

from deepks.descriptor import is_ghost_atom


class DeePHFCapabilityError(ValueError):
    """Raised when a reference is outside the declared DeePHF domain."""


_MODULE_EXECUTION_HOOK_FIELDS = (
    ("forward-pre", "_forward_pre_hooks"),
    ("forward", "_forward_hooks"),
    ("backward-pre", "_backward_pre_hooks"),
    ("backward", "_backward_hooks"),
)
_GLOBAL_MODULE_EXECUTION_HOOK_FIELDS = (
    ("global-forward-pre", "_global_forward_pre_hooks"),
    ("global-forward", "_global_forward_hooks"),
    ("global-backward-pre", "_global_backward_pre_hooks"),
    ("global-backward", "_global_backward_hooks"),
)
_MODULE_CONTAINER_FIELDS = frozenset({"_parameters", "_buffers", "_modules"})


def _qualified_type(value) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _update_model_tensor_fingerprint(digest, tensor: torch.Tensor) -> None:
    digest.update(str(tensor.layout).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    if tensor.device.type == "meta":
        raise DeePHFCapabilityError(
            "the force correction model cannot use meta-device state"
        )
    try:
        value = tensor.detach().cpu()
        if value.layout != torch.strided:
            value = value.to_dense()
        flat_value = torch.empty(value.numel(), dtype=value.dtype, device="cpu")
        flat_value.copy_(value.reshape(-1))
        digest.update(flat_value.view(torch.uint8).numpy().tobytes())
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the force correction model tensor could not be fingerprinted: {error}"
        ) from error


def _update_model_metadata_fingerprint(digest, value, active_objects) -> None:
    digest.update(_qualified_type(value).encode("utf-8"))
    if value is None:
        return
    if isinstance(value, (bool, int, str, bytes)):
        digest.update(repr(value).encode("utf-8"))
        return
    if isinstance(value, float):
        digest.update(value.hex().encode("ascii"))
        return
    if isinstance(value, np.generic):
        _update_model_metadata_fingerprint(digest, value.item(), active_objects)
        return
    if isinstance(value, torch.Tensor):
        _update_model_tensor_fingerprint(digest, value)
        return
    if isinstance(value, np.ndarray):
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        if value.dtype.hasobject:
            digest.update(repr(value.tolist()).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, random.Random):
        digest.update(repr(value.getstate()).encode("utf-8"))
        return
    if isinstance(value, np.random.Generator):
        _update_model_metadata_fingerprint(
            digest,
            value.bit_generator.state,
            active_objects,
        )
        return
    if isinstance(value, np.random.RandomState):
        _update_model_metadata_fingerprint(
            digest,
            value.get_state(),
            active_objects,
        )
        return
    if isinstance(value, torch.Generator):
        digest.update(str(value.device).encode("utf-8"))
        _update_model_tensor_fingerprint(digest, value.get_state())
        return
    if callable(value):
        digest.update(str(getattr(value, "__module__", "")).encode("utf-8"))
        digest.update(
            str(getattr(value, "__qualname__", repr(value))).encode("utf-8")
        )
        return
    identity = id(value)
    if identity in active_objects:
        digest.update(b"<recursive>")
        return
    active_objects.add(identity)
    try:
        if isinstance(value, Mapping):
            ordered_items = sorted(
                value.items(),
                key=lambda item: (_qualified_type(item[0]), repr(item[0])),
            )
            for key, item in ordered_items:
                _update_model_metadata_fingerprint(digest, key, active_objects)
                _update_model_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                _update_model_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, (set, frozenset)):
            for item in sorted(
                value,
                key=lambda item: (_qualified_type(item), repr(item)),
            ):
                _update_model_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, torch.nn.Module):
            digest.update(_qualified_type(value).encode("utf-8"))
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in sorted(attributes.items()):
                digest.update(name.encode("utf-8"))
                _update_model_metadata_fingerprint(digest, item, active_objects)
            return
        digest.update(repr(value).encode("utf-8"))
    finally:
        active_objects.remove(identity)


def validate_reference(reference):
    """Validate the molecular real-orbital integer-occupation RHF contract."""
    if type(reference) is not scf.hf.RHF:
        raise DeePHFCapabilityError(
            "DeePHF requires an undecorated native pyscf.scf.hf.RHF reference"
        )
    if not reference.converged:
        raise DeePHFCapabilityError("the RHF reference must be converged")
    mol = reference.mol
    if type(mol) is not gto.mole.Mole:
        raise DeePHFCapabilityError(
            "the RHF reference must use a native molecular pyscf.gto.Mole"
        )
    if mol.spin != 0:
        raise DeePHFCapabilityError("the RHF reference must have spin zero")
    if mol.symmetry:
        raise DeePHFCapabilityError(
            "the RHF reference must not use symmetry-constrained occupations"
        )
    if mol.cart:
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires spherical AO functions"
        )
    if getattr(mol, "_ecp", None) or mol.has_ecp():
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires an all-electron reference"
        )
    if getattr(mol, "_pseudo", None):
        raise DeePHFCapabilityError(
            "the initial DeePHF contract does not support pseudopotentials"
        )
    if float(getattr(mol, "omega", 0.0)) != 0.0:
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires the full Coulomb interaction"
        )
    if getattr(mol, "nucmod", None):
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires point nuclei"
        )
    ghost_indices = [
        atom_index
        for atom_index in range(mol.natm)
        if is_ghost_atom(mol, atom_index)
    ]
    if ghost_indices:
        raise DeePHFCapabilityError(
            f"the initial DeePHF contract requires real atoms; ghost indices: {ghost_indices}"
        )
    decorated_attributes = {
        "density fitting": "with_df",
        "solvent": "with_solvent",
        "X2C": "with_x2c",
        "QM/MM": "mm_mol",
        "dispersion": "disp",
        "penalty": "penalties",
    }
    active_decorations = [
        name
        for name, attribute in decorated_attributes.items()
        if getattr(reference, attribute, None)
    ]
    if active_decorations:
        raise DeePHFCapabilityError(
            "the RHF reference has unsupported decorations: "
            + ", ".join(active_decorations)
        )
    custom_hooks = sorted(
        name
        for name, value in reference.__dict__.items()
        if name != "mol" and callable(value)
    )
    if custom_hooks:
        raise DeePHFCapabilityError(
            "the RHF reference has unsupported instance hooks: "
            + ", ".join(custom_hooks)
        )
    molecule_hooks = sorted(
        name for name, value in mol.__dict__.items() if callable(value)
    )
    if molecule_hooks:
        raise DeePHFCapabilityError(
            "the RHF molecule has unsupported instance hooks: "
            + ", ".join(molecule_hooks)
        )
    if reference.mo_coeff is None or reference.mo_energy is None:
        raise DeePHFCapabilityError("the RHF reference orbital state is incomplete")
    if reference.mo_occ is None:
        raise DeePHFCapabilityError("the RHF reference occupations are missing")
    mo_coeff = np.asarray(reference.mo_coeff)
    mo_energy = np.asarray(reference.mo_energy)
    occupations = np.asarray(reference.mo_occ)
    if any(
        np.iscomplexobj(value)
        for value in (mo_coeff, mo_energy, occupations)
    ):
        raise DeePHFCapabilityError("the RHF orbitals must be real")
    if any(
        value.dtype != np.dtype(np.float64)
        for value in (mo_coeff, mo_energy, occupations)
    ):
        raise DeePHFCapabilityError(
            "the RHF orbital state must use numpy.float64"
        )
    if not all(
        np.isfinite(value).all()
        for value in (mo_coeff, mo_energy, occupations)
    ):
        raise DeePHFCapabilityError("the RHF orbital state must be finite")
    if mo_coeff.shape != (mol.nao, mol.nao):
        raise DeePHFCapabilityError(
            "the RHF response requires a complete square MO coefficient matrix"
        )
    if mo_energy.shape != (mo_coeff.shape[1],):
        raise DeePHFCapabilityError("the RHF orbital energy shape is invalid")
    if occupations.shape != mo_energy.shape:
        raise DeePHFCapabilityError("the RHF occupation shape is invalid")
    occupations = np.asarray(reference.mo_occ)
    if not np.all(np.isin(occupations, (0.0, 2.0))):
        raise DeePHFCapabilityError(
            "the RHF occupations must be integer closed-shell occupations"
        )
    if not np.isclose(occupations.sum(), mol.nelectron, rtol=0.0, atol=1.0e-12):
        raise DeePHFCapabilityError(
            "the RHF occupations do not match the molecular electron count"
        )
    occupied_count = mol.nelectron // 2
    expected_occupations = np.zeros_like(occupations)
    expected_occupations[:occupied_count] = 2.0
    if not np.array_equal(occupations, expected_occupations):
        raise DeePHFCapabilityError(
            "the initial RHF force contract requires the Aufbau ground-state root"
        )
    if not np.isfinite(reference.e_tot):
        raise DeePHFCapabilityError("the RHF reference energy must be finite")
    try:
        overlap = np.asarray(reference.get_ovlp())
        hcore = np.asarray(reference.get_hcore())
        density = np.asarray(reference.make_rdm1())
        effective_potential = np.asarray(reference.get_veff(mol, density))
        direct_coulomb, direct_exchange = scf.hf.get_jk(
            mol,
            density,
            hermi=1,
        )
        direct_effective_potential = np.asarray(
            direct_coulomb - 0.5 * direct_exchange
        )
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the RHF reference matrices could not be evaluated: {error}"
        ) from error
    if any(
        np.iscomplexobj(value)
        for value in (
            overlap,
            hcore,
            density,
            effective_potential,
            direct_effective_potential,
        )
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must be real")
    if any(
        value.dtype != np.dtype(np.float64)
        for value in (
            overlap,
            hcore,
            density,
            effective_potential,
            direct_effective_potential,
        )
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must use numpy.float64")
    if not all(
        np.isfinite(value).all()
        for value in (
            overlap,
            hcore,
            density,
            effective_potential,
            direct_effective_potential,
        )
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must be finite")
    expected_ao_shape = (mol.nao, mol.nao)
    if any(
        value.shape != expected_ao_shape
        for value in (
            overlap,
            hcore,
            density,
            effective_potential,
            direct_effective_potential,
        )
    ):
        raise DeePHFCapabilityError("the RHF AO matrix shape is invalid")
    interaction_error = np.max(
        np.abs(effective_potential - direct_effective_potential),
        initial=0.0,
    )
    if interaction_error > 1.0e-10:
        raise DeePHFCapabilityError(
            "the RHF two-electron interaction does not match the native "
            f"molecular integrals: residual {interaction_error:.3e}"
        )
    overlap_eigenvalues = np.linalg.eigvalsh(overlap)
    if overlap_eigenvalues[0] <= 1.0e-10:
        raise DeePHFCapabilityError(
            "the RHF AO overlap is singular or ill conditioned"
        )
    orthonormality_error = np.max(
        np.abs(mo_coeff.T @ overlap @ mo_coeff - np.eye(mo_coeff.shape[1]))
    )
    if orthonormality_error > 1.0e-8:
        raise DeePHFCapabilityError(
            "the RHF orbitals violate AO-metric orthonormality: "
            f"{orthonormality_error:.3e}"
        )
    electron_count = np.einsum("ij,ji->", density, overlap)
    if not np.isclose(electron_count, mol.nelectron, rtol=0.0, atol=1.0e-8):
        raise DeePHFCapabilityError(
            "the RHF AO density has an inconsistent electron count: "
            f"{electron_count:.12g}"
        )
    fock = hcore + direct_effective_potential
    canonical_residual = fock @ mo_coeff - overlap @ (
        mo_coeff * mo_energy
    )
    maximum_canonical_residual = np.max(
        np.abs(canonical_residual),
        initial=0.0,
    )
    if maximum_canonical_residual > 1.0e-7:
        raise DeePHFCapabilityError(
            "the stored RHF orbitals and energies do not satisfy the canonical "
            f"SCF equations: residual {maximum_canonical_residual:.3e}"
        )
    recomputed_energy = (
        0.5 * np.einsum("ij,ji->", density, hcore + fock)
        + mol.energy_nuc()
    )
    if not np.isclose(
        recomputed_energy,
        reference.e_tot,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise DeePHFCapabilityError(
            "the stored RHF total energy is inconsistent with its AO state: "
            f"{reference.e_tot:.12g} != {recomputed_energy:.12g}"
        )
    if not np.isfinite(mol.atom_coords(unit="Bohr")).all():
        raise DeePHFCapabilityError("the molecular geometry must be finite")
    return reference


def _metadata_signature(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_signature(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_force_model(model):
    """Require stable evaluation mode and hook-free execution for force inference."""
    if model is None:
        return None
    if not isinstance(model, torch.nn.Module):
        raise DeePHFCapabilityError(
            "the force correction model must be a torch.nn.Module or None"
        )
    try:
        modules = tuple(model.named_modules(remove_duplicate=False))
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the force correction model modules could not be inspected: {error}"
        ) from error
    training_modules = [
        name or "<root>"
        for name, module in modules
        if module.training is not False
    ]
    if training_modules:
        raise DeePHFCapabilityError(
            "the force correction model must remain in evaluation mode; "
            f"training modules: {', '.join(training_modules)}"
        )
    active_hooks = []
    for name, module in modules:
        module_name = name or "<root>"
        for hook_name, field_name in _MODULE_EXECUTION_HOOK_FIELDS:
            try:
                registry = getattr(module, field_name)
            except Exception as error:
                raise DeePHFCapabilityError(
                    "the force correction model hook registry could not be "
                    f"inspected: {module_name}:{hook_name}: {error}"
                ) from error
            if not isinstance(registry, Mapping):
                raise DeePHFCapabilityError(
                    "the force correction model hook registry is invalid: "
                    f"{module_name}:{hook_name}"
                )
            if registry:
                active_hooks.append(f"{module_name}:{hook_name}")
    global_module_hooks = torch.nn.modules.module
    for hook_name, field_name in _GLOBAL_MODULE_EXECUTION_HOOK_FIELDS:
        try:
            registry = getattr(global_module_hooks, field_name)
        except Exception as error:
            raise DeePHFCapabilityError(
                "the global force-model hook registry could not be inspected: "
                f"{hook_name}: {error}"
            ) from error
        if not isinstance(registry, Mapping):
            raise DeePHFCapabilityError(
                f"the global force-model hook registry is invalid: {hook_name}"
            )
        if registry:
            active_hooks.append(hook_name)
    if active_hooks:
        raise DeePHFCapabilityError(
            "the force correction model cannot contain module execution hooks; "
            f"active hooks: {', '.join(active_hooks)}"
        )
    return model


def force_model_fingerprint(model) -> str:
    """Bind force-model structure, semantic attributes, parameters, and buffers."""
    validate_force_model(model)
    digest = hashlib.sha256()
    if model is None:
        digest.update(b"deepks.deephf.none-force-correction-model")
        return digest.hexdigest()
    try:
        modules = tuple(model.named_modules(remove_duplicate=False))
        parameters = tuple(model.named_parameters(remove_duplicate=False))
        buffers = tuple(model.named_buffers(remove_duplicate=False))
        state = model.state_dict()
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the force correction model state could not be enumerated: {error}"
        ) from error
    for name, module in modules:
        digest.update(b"module\0")
        digest.update(name.encode("utf-8"))
        digest.update(_qualified_type(module).encode("utf-8"))
        for attribute_name, value in sorted(module.__dict__.items()):
            if attribute_name in _MODULE_CONTAINER_FIELDS:
                continue
            digest.update(attribute_name.encode("utf-8"))
            _update_model_metadata_fingerprint(digest, value, set())
    for name, parameter in parameters:
        digest.update(b"parameter\0")
        digest.update(name.encode("utf-8"))
        digest.update(repr(bool(parameter.requires_grad)).encode("ascii"))
        _update_model_tensor_fingerprint(digest, parameter)
    for name, buffer in buffers:
        digest.update(b"buffer\0")
        digest.update(name.encode("utf-8"))
        _update_model_tensor_fingerprint(digest, buffer)
    if not isinstance(state, Mapping):
        raise DeePHFCapabilityError(
            "the force correction model state_dict must be a mapping"
        )
    for name, value in sorted(state.items()):
        digest.update(b"state\0")
        digest.update(str(name).encode("utf-8"))
        _update_model_metadata_fingerprint(digest, value, set())
    return digest.hexdigest()


def force_rng_fingerprints() -> dict[str, str]:
    """Snapshot global Python, NumPy, and initialized Torch RNG states."""
    result = {}
    python_digest = hashlib.sha256(repr(random.getstate()).encode("utf-8"))
    result["Python"] = python_digest.hexdigest()
    numpy_digest = hashlib.sha256()
    numpy_state = np.random.get_state()
    for value in numpy_state:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            numpy_digest.update(array.dtype.str.encode("ascii"))
            numpy_digest.update(repr(array.shape).encode("ascii"))
            numpy_digest.update(array.tobytes())
        else:
            numpy_digest.update(repr(value).encode("utf-8"))
    result["NumPy"] = numpy_digest.hexdigest()
    torch_cpu_state = torch.random.get_rng_state()
    torch_cpu_digest = hashlib.sha256(
        torch_cpu_state.cpu().numpy().tobytes()
    )
    result["Torch CPU"] = torch_cpu_digest.hexdigest()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_digest = hashlib.sha256()
        for state in torch.cuda.get_rng_state_all():
            cuda_digest.update(state.cpu().numpy().tobytes())
        result["Torch CUDA"] = cuda_digest.hexdigest()
    return result


def validate_model(model, projector_basis, descriptor_features: int):
    """Validate the strict double-precision scalar correction-model contract."""
    if model is None:
        return None
    if not isinstance(model, torch.nn.Module):
        raise DeePHFCapabilityError(
            "the DeePHF correction model must be a torch.nn.Module or None"
        )
    tensors = list(model.parameters()) + list(model.buffers())
    for tensor in tensors:
        if tensor.is_complex():
            raise DeePHFCapabilityError("the correction model must be real")
        if tensor.is_floating_point() and tensor.dtype != torch.float64:
            raise DeePHFCapabilityError(
                "the correction model must use torch.float64"
            )
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise DeePHFCapabilityError(
                "the correction model parameters and buffers must be finite"
            )
    input_dimension = getattr(model, "input_dim", descriptor_features)
    if input_dimension != descriptor_features:
        raise DeePHFCapabilityError(
            "the correction model input dimension does not match the descriptor: "
            f"{input_dimension} != {descriptor_features}"
        )
    model_basis = getattr(model, "_pbas", None)
    if (
        model_basis is not None
        and _metadata_signature(model_basis) != _metadata_signature(projector_basis)
    ):
        raise DeePHFCapabilityError(
            "the correction model projector metadata does not match projector_basis"
        )
    return model


def validate_model_output(model, descriptor_values: torch.Tensor) -> torch.Tensor:
    """Evaluate and validate one real finite scalar correction energy."""
    if model is None:
        return torch.zeros((), dtype=torch.float64)
    try:
        reference_tensor = next(model.parameters())
    except StopIteration:
        try:
            reference_tensor = next(model.buffers())
        except StopIteration:
            reference_tensor = torch.empty((), dtype=torch.float64)
    try:
        output = model(descriptor_values.to(reference_tensor))
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the correction model evaluation failed: {error}"
        ) from error
    if not isinstance(output, torch.Tensor):
        raise DeePHFCapabilityError("the correction model output must be a tensor")
    if output.is_complex():
        raise DeePHFCapabilityError("the correction model output must be real")
    if output.dtype != torch.float64:
        raise DeePHFCapabilityError(
            "the correction model output must use torch.float64"
        )
    if output.numel() != 1:
        raise DeePHFCapabilityError(
            "the correction model must produce exactly one scalar energy; "
            f"received shape {tuple(output.shape)}"
        )
    if not torch.isfinite(output).all():
        raise DeePHFCapabilityError("the correction model output must be finite")
    return output
