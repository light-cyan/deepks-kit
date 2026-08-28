"""GPU-native perturbative DeePHF energy and analytic-gradient methods."""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Real

import numpy as np
import torch

from deepks.descriptor import (
    AtomicDensityDescriptor,
    spin_summed_ao_density,
    validate_differentiability,
)
from deepks.gpu import (
    DEFAULT_CUDA_DEVICE,
    as_numpy,
    cupy_from_torch,
    require_cuda_device,
    torch_from_array,
)
from deepks.model.model import (
    CorrNet,
    corrnet_is_shell_permutation_invariant,
)

from .capabilities import (
    DeePHFCapabilityError,
    validate_force_model,
    validate_model,
    validate_model_output,
)
from .driver import validate_atom_indices


GPU_REFERENCE_TYPES = {
    "gpu4pyscf.scf.hf.RHF": "rhf",
    "gpu4pyscf.scf.uhf.UHF": "uhf",
    "gpu4pyscf.dft.rks.RKS": "rks",
    "gpu4pyscf.dft.uks.UKS": "uks",
}
RESTRICTED_FAMILIES = frozenset(("rhf", "rks"))
SUPPORTED_RESPONSE_OPTIONS = frozenset(
    (
        "cphf_tolerance",
        "residual_tolerance",
        "invariant_tolerance",
        "orbital_gap_tolerance",
        "max_cycle",
        "max_refinement_cycles",
        "level_shift",
        "coordinate_block_size",
    )
)


def gpu_reference_family(reference) -> str | None:
    """Return the family of an exact GPU4PySCF molecular reference."""
    qualified_name = (
        f"{type(reference).__module__}.{type(reference).__qualname__}"
    )
    return GPU_REFERENCE_TYPES.get(qualified_name)


def is_gpu_reference(reference) -> bool:
    """Return whether a reference belongs to the supported GPU-native tier."""
    return gpu_reference_family(reference) is not None


def _validate_response_options(options) -> dict:
    if options is None:
        return {}
    if not isinstance(options, dict):
        try:
            options = dict(options)
        except (TypeError, ValueError) as error:
            raise TypeError("response_options must be a mapping") from error
    unknown = sorted(set(options) - SUPPORTED_RESPONSE_OPTIONS)
    if unknown:
        raise ValueError(
            "unsupported GPU direct-response options: " + ", ".join(unknown)
        )
    return options


def _finite_real_scalar(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise DeePHFCapabilityError(f"{name} must be a real scalar") from error
    result = float(value)
    if not np.isfinite(result):
        raise DeePHFCapabilityError(f"{name} must be finite")
    return result


class GPUDeePHF:
    """Evaluate DeePHF energy and its complete direct derivative on CUDA."""

    reference_family = "rhf"

    def __init__(
        self,
        reference,
        model,
        projector_basis=None,
        device=DEFAULT_CUDA_DEVICE,
        response_options=None,
        adjoint_options=None,
    ):
        family = gpu_reference_family(reference)
        if family != self.reference_family:
            raise DeePHFCapabilityError(
                f"{type(self).__name__} requires an exact GPU4PySCF "
                f"{self.reference_family.upper()} reference"
            )
        if not bool(getattr(reference, "converged", False)):
            raise DeePHFCapabilityError("the GPU4PySCF reference must be converged")
        self.device = require_cuda_device(device)
        if isinstance(model, str):
            model = CorrNet.load(model, strict=True).double()
        if isinstance(model, torch.nn.Module):
            model = model.to(self.device).eval()
        self.reference = reference
        self.model = model
        if projector_basis is None:
            projector_basis = getattr(model, "_pbas", None)
        self._descriptor = AtomicDensityDescriptor(
            reference.mol,
            projector_basis,
            device=self.device,
        )
        validate_model(
            self.model,
            self._descriptor.projector_basis,
            self._descriptor.n_features,
        )
        self.response_options = _validate_response_options(response_options)
        self.adjoint_options = {} if adjoint_options is None else dict(adjoint_options)
        self.e_base = None
        self.e_corr = None
        self.e_tot = None
        self.converged = True
        self._workspace = None
        self._descriptor_values = None
        self._sensitivity = None
        self._density = None
        self._spin_density = None
        self._gradient = None
        self._gradient_options = None
        self._operation_counts = {}

    @property
    def mol(self):
        return self.reference.mol

    @property
    def n_descriptor_atoms(self) -> int:
        return self._descriptor.n_descriptor_atoms

    @property
    def n_descriptor_features(self) -> int:
        return self._descriptor.n_features

    @property
    def operation_counts(self) -> dict:
        return dict(self._operation_counts)

    @contextmanager
    def _controlled_calculation(self):
        from .workflow import _reference_state_fingerprint

        initial_state = _reference_state_fingerprint(self.reference)
        try:
            yield self
        finally:
            final_state = _reference_state_fingerprint(self.reference)
            if final_state != initial_state:
                raise DeePHFCapabilityError(
                    "the DeePHF scientific state changed during the controlled calculation"
                )

    def _descriptor_rank_bound(self) -> int:
        occupations = as_numpy(self.reference.mo_occ)
        occupied_count = int(np.count_nonzero(occupations > 0))
        if self.reference_family in RESTRICTED_FAMILIES:
            return occupied_count
        return min(int(self.mol.nao), occupied_count)

    def _element_constant(self) -> float:
        if self.model is None:
            return 0.0
        getter = getattr(self.model, "get_elem_const", None)
        if getter is None:
            return 0.0
        value = getter([int(charge) for charge in self.mol.atom_charges()])
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise DeePHFCapabilityError(
                    "the correction model element constant must be scalar"
                )
            value = value.detach().item()
        else:
            array = np.asarray(value)
            if array.shape != ():
                raise DeePHFCapabilityError(
                    "the correction model element constant must be scalar"
                )
            value = array.item()
        return _finite_real_scalar(value, "correction model element constant")

    def _evaluate_model(self) -> None:
        if self._workspace is not None:
            return
        spin_density = self.reference.make_rdm1()
        total_density = spin_summed_ao_density(spin_density)
        density = torch_from_array(total_density, device=self.device)
        self._density = density
        if getattr(spin_density, "ndim", None) == 3:
            self._spin_density = torch_from_array(spin_density, device=self.device)
        self._workspace = self._descriptor.derivative_workspace(total_density)
        descriptor_values = self._workspace.descriptor_values.detach()
        descriptor_values.requires_grad_(self.model is not None)
        if self.model is None:
            model_output = torch.zeros((), dtype=torch.float64, device=self.device)
            sensitivity = torch.zeros_like(descriptor_values)
        else:
            validate_force_model(self.model)
            with torch.enable_grad():
                model_output = validate_model_output(self.model, descriptor_values)
            if model_output.requires_grad:
                (sensitivity,) = torch.autograd.grad(
                    model_output,
                    descriptor_values,
                    torch.ones_like(model_output),
                    allow_unused=True,
                )
            else:
                sensitivity = None
            if sensitivity is None:
                sensitivity = torch.zeros_like(descriptor_values)
        if (
            sensitivity.shape != descriptor_values.shape
            or sensitivity.dtype != torch.float64
            or sensitivity.is_complex()
            or not bool(torch.isfinite(sensitivity).all().item())
        ):
            raise DeePHFCapabilityError(
                "the correction model descriptor sensitivity is invalid"
            )
        if self.model is not None:
            validate_differentiability(
                descriptor_values.detach().cpu().numpy(),
                self._descriptor.shell_sizes,
                self._descriptor_rank_bound(),
                sensitivity.detach().cpu().numpy(),
                symmetric_function=corrnet_is_shell_permutation_invariant(
                    self.model
                ),
            )
        self._descriptor_values = descriptor_values
        self._sensitivity = sensitivity.detach()
        self.e_base = _finite_real_scalar(
            self.reference.e_tot,
            "GPU4PySCF reference energy",
        )
        self.e_corr = _finite_real_scalar(
            model_output.detach().item() + self._element_constant(),
            "DeePHF correction energy",
        )
        self.e_tot = _finite_real_scalar(
            self.e_base + self.e_corr,
            "DeePHF total energy",
        )

    def kernel(self) -> float:
        """Return the perturbatively corrected total energy in Hartree."""
        self._evaluate_model()
        return self.e_tot

    def ao_density(self) -> np.ndarray:
        self._evaluate_model()
        return as_numpy(self._density)

    def descriptor(self) -> np.ndarray:
        self._evaluate_model()
        return self._descriptor_values.detach().cpu().numpy().copy()

    def correction_sensitivity(self) -> np.ndarray:
        self._evaluate_model()
        return self._sensitivity.detach().cpu().numpy().copy()

    def _solver_controls(self, options) -> tuple[float, int, float]:
        tolerance = float(
            options.get(
                "cphf_tolerance",
                getattr(self.reference, "conv_tol_cpscf", 1.0e-12),
            )
        )
        max_cycle = int(options.get("max_cycle", 100))
        level_shift = float(options.get("level_shift", 0.0))
        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ValueError("cphf_tolerance must be finite and positive")
        if max_cycle <= 0:
            raise ValueError("max_cycle must be positive")
        if level_shift < 0.0 or not np.isfinite(level_shift):
            raise ValueError("level_shift must be finite and nonnegative")
        return tolerance, max_cycle, level_shift

    def _overlap_mo_derivative_restricted(self, coefficient, occupation, shape):
        import cupy
        from gpu4pyscf.pbc.gto.int1e import int1e_ipovlp

        occupied = coefficient[:, occupation > 0]
        derivative = -int1e_ipovlp(self.mol)
        result = cupy.empty(shape, dtype=cupy.float64)
        for atom_index, (_shell0, _shell1, ao0, ao1) in enumerate(
            self.mol.aoslice_by_atom()
        ):
            ao_derivative = cupy.zeros(
                (3, self.mol.nao, self.mol.nao), dtype=cupy.float64
            )
            ao_derivative[:, ao0:ao1] += derivative[:, ao0:ao1]
            ao_derivative[:, :, ao0:ao1] += derivative[:, ao0:ao1].transpose(
                0, 2, 1
            )
            result[atom_index] = cupy.einsum(
                "mp,xmn,ni->xpi",
                coefficient,
                ao_derivative,
                occupied,
            )
        return result

    def _overlap_mo_derivative_unrestricted(self, coefficient, occupation, shapes):
        return tuple(
            self._overlap_mo_derivative_restricted(
                coefficient[spin], occupation[spin], shapes[spin]
            )
            for spin in range(2)
        )

    @staticmethod
    def _restricted_density_response(coefficient, occupation, mo_response):
        import cupy

        occupied = coefficient[:, occupation > 0]
        coefficient_response = cupy.einsum(
            "mp,bxpi->bxmi", coefficient, mo_response
        )
        one_sided = cupy.einsum(
            "bxmi,ni->bxmn", coefficient_response, occupied
        )
        return 2.0 * (one_sided + one_sided.transpose(0, 1, 3, 2))

    @staticmethod
    def _unrestricted_density_response(coefficient, occupation, mo_response):
        import cupy

        responses = []
        for spin in range(2):
            occupied = coefficient[spin][:, occupation[spin] > 0]
            coefficient_response = cupy.einsum(
                "mp,bxpi->bxmi", coefficient[spin], mo_response[spin]
            )
            one_sided = cupy.einsum(
                "bxmi,ni->bxmn", coefficient_response, occupied
            )
            responses.append(one_sided + one_sided.transpose(0, 1, 3, 2))
        return tuple(responses)

    def _density_response(self, options):
        import cupy

        tolerance, max_cycle, level_shift = self._solver_controls(options)
        coefficient = self.reference.mo_coeff
        occupation = self.reference.mo_occ
        hessian = self.reference.Hessian()
        h1mo = hessian.make_h1(coefficient, occupation)
        vind = hessian.gen_vind(coefficient, occupation)
        if self.reference_family in RESTRICTED_FAMILIES:
            from gpu4pyscf.scf import cphf

            perturbation_shape = h1mo.shape[:2]
            overlap = self._overlap_mo_derivative_restricted(
                coefficient,
                occupation,
                h1mo.shape,
            )
            mo_response, _orbital_energy_response = cphf.solve(
                vind,
                self.reference.mo_energy,
                occupation,
                h1mo.reshape(-1, *h1mo.shape[-2:]),
                overlap.reshape(-1, *overlap.shape[-2:]),
                max_cycle=max_cycle,
                tol=tolerance,
                verbose=self.reference.verbose,
                level_shift=level_shift,
            )
            mo_response = mo_response.reshape(
                *perturbation_shape, *mo_response.shape[-2:]
            )
            result = self._restricted_density_response(
                coefficient, occupation, mo_response
            )
        else:
            from gpu4pyscf.scf import ucphf

            overlap = self._overlap_mo_derivative_unrestricted(
                coefficient,
                occupation,
                (h1mo[0].shape, h1mo[1].shape),
            )
            perturbation_shape = h1mo[0].shape[:2]
            mo_response, _orbital_energy_response = ucphf.solve(
                vind,
                self.reference.mo_energy,
                occupation,
                tuple(
                    value.reshape(-1, *value.shape[-2:]) for value in h1mo
                ),
                tuple(
                    value.reshape(-1, *value.shape[-2:]) for value in overlap
                ),
                max_cycle=max_cycle,
                tol=tolerance,
                verbose=self.reference.verbose,
                level_shift=level_shift,
            )
            mo_response = tuple(
                value.reshape(*perturbation_shape, *value.shape[-2:])
                for value in mo_response
            )
            spin_response = self._unrestricted_density_response(
                coefficient, occupation, mo_response
            )
            result = spin_response[0] + spin_response[1]
        if (
            result.shape
            != (self.mol.natm, 3, self.mol.nao, self.mol.nao)
            or result.dtype != cupy.float64
            or not bool(cupy.isfinite(result).all().item())
        ):
            raise DeePHFCapabilityError(
                "the GPU coupled-perturbed density response is invalid"
            )
        self._operation_counts["gpu_direct_response_solves"] = 1
        return result

    def _analytic_gradient(self, atom_indices=None, response_options=None) -> np.ndarray:
        self._evaluate_model()
        effective_options = {
            **self.response_options,
            **_validate_response_options(response_options),
        }
        option_key = tuple(sorted(effective_options.items()))
        if self._gradient is None or self._gradient_options != option_key:
            import cupy

            native = cupy.asarray(
                self.reference.nuc_grad_method().kernel(), dtype=cupy.float64
            )
            if bool(torch.count_nonzero(self._sensitivity).item()):
                explicit, objective = self._workspace.correction_derivatives_tensor(
                    self._sensitivity
                )
                response_density = self._density_response(effective_options)
                response = cupy.einsum(
                    "ij,bxij->bx", cupy_from_torch(objective), response_density
                )
                total = native + cupy_from_torch(explicit) + response
            else:
                total = native
            if (
                total.shape != (self.mol.natm, 3)
                or total.dtype != cupy.float64
                or not bool(cupy.isfinite(total).all().item())
            ):
                raise DeePHFCapabilityError("the GPU DeePHF gradient is invalid")
            self._gradient = total
            self._gradient_options = option_key
        selected = validate_atom_indices(self.mol, atom_indices)
        values = self._gradient if selected is None else self._gradient[list(selected)]
        return np.array(as_numpy(values), dtype=np.float64, copy=True, order="C")

    def nuc_grad_method(
        self,
        *,
        backend="direct",
        retain_details=False,
        **backend_options,
    ):
        if backend != "direct":
            raise DeePHFCapabilityError(
                "GPU-native DeePHF currently provides the direct analytic-gradient backend"
            )
        if backend_options:
            merged = {**self.response_options, **backend_options}
            _validate_response_options(merged)
        return GPUDeePHFGradients(self, options=backend_options)

    def gradient(self, *, backend="direct", **backend_options) -> np.ndarray:
        return self.nuc_grad_method(
            backend=backend, **backend_options
        ).kernel()

    def forces(self, *, backend="direct", **backend_options) -> np.ndarray:
        return -self.gradient(backend=backend, **backend_options)


class GPUUHFDeePHF(GPUDeePHF):
    """GPU-native perturbative DeePHF around an unrestricted HF reference."""

    reference_family = "uhf"


class GPURKSDeePHF(GPUDeePHF):
    """GPU-native perturbative DeePHF around a restricted KS reference."""

    reference_family = "rks"


class GPUUKSDeePHF(GPUDeePHF):
    """GPU-native perturbative DeePHF around an unrestricted KS reference."""

    reference_family = "uks"


GPU_METHOD_CLASSES = {
    "rhf": GPUDeePHF,
    "uhf": GPUUHFDeePHF,
    "rks": GPURKSDeePHF,
    "uks": GPUUKSDeePHF,
}


class GPUDeePHFGradients:
    """Complete GPU-native DeePHF direct analytic gradient."""

    backend = "direct"

    def __init__(self, method, *, options=None):
        if not isinstance(method, GPUDeePHF):
            raise TypeError("GPUDeePHFGradients requires a GPUDeePHF method")
        self.base = method
        self.response_options = dict(options or {})
        self.mol = method.mol
        self.de = None

    def kernel(self, atmlst=None) -> np.ndarray:
        self.de = self.base._analytic_gradient(
            atmlst,
            response_options=self.response_options,
        )
        return self.de.copy()

    def run(self, atmlst=None):
        self.kernel(atmlst=atmlst)
        return self

    def as_scanner(self, **scanner_options):
        from .gpu_scanner import GPUDeePHFGradientScanner

        return GPUDeePHFGradientScanner.from_method(
            self.base,
            backend=self.backend,
            backend_options=self.response_options,
            **scanner_options,
        )


__all__ = [
    "GPUDeePHF",
    "GPUDeePHFGradients",
    "GPURKSDeePHF",
    "GPUUHFDeePHF",
    "GPUUKSDeePHF",
    "GPU_METHOD_CLASSES",
    "gpu_reference_family",
    "is_gpu_reference",
]
