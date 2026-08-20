"""Perturbative DeePHF energy method composed around a native reference."""

from collections.abc import Mapping
import hashlib

import numpy as np
import torch

from deepks.descriptor import (
    AtomicDensityDescriptor,
    DescriptorDiagnostics,
    spin_summed_ao_density,
    validate_differentiability,
)
from deepks.model.model import CorrNet

from .capabilities import (
    DeePHFCapabilityError,
    force_model_fingerprint,
    force_rng_fingerprints,
    validate_force_model,
    validate_model,
    validate_model_output,
)
from .pyscf_rhf import (
    RHFAdjoint,
    RHFAdjointAdapter,
    RHFAdjointDiagnostics,
    RHFAdjointError,
    RHFResponse,
    RHFResponseAdapter,
    RHFResponseDiagnostics,
    RHFResponseError,
    adjoint_integrity_fingerprint,
    molecule_science_fingerprint,
    reference_fingerprint,
    response_integrity_fingerprint,
    validate_reference,
)


_DIRECT_RESPONSE_OPTIONS = frozenset(
    {
        "cphf_tolerance",
        "residual_tolerance",
        "invariant_tolerance",
        "orbital_gap_tolerance",
        "max_cycle",
        "max_refinement_cycles",
        "level_shift",
        "operator_stability_tolerance",
        "operator_condition_tolerance",
        "operator_symmetry_tolerance",
        "operator_dimension_limit",
    }
)
_ZVECTOR_OPTIONS = frozenset(
    {
        "residual_tolerance",
        "orbital_gap_tolerance",
        "operator_stability_tolerance",
        "operator_condition_tolerance",
        "operator_symmetry_tolerance",
        "operator_dimension_limit",
        "objective_symmetry_tolerance",
    }
)
_FORCE_DESCRIPTOR_FD_STEP = 1.0e-5
_FORCE_DESCRIPTOR_FD_ATOL = 2.0e-7
_FORCE_DESCRIPTOR_FD_RTOL = 2.0e-5


def _validated_backend_options(base_options, override_options, allowed, backend):
    options = {**base_options, **override_options}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            f"unsupported {backend} backend options: {', '.join(unknown)}"
        )
    return options


def _immutable_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _update_science_digest(digest, value) -> None:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        return
    if isinstance(value, np.generic):
        _update_science_digest(digest, value.item())
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_science_digest(digest, key)
            _update_science_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _update_science_digest(digest, item)
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
        return
    raise DeePHFCapabilityError(
        "cannot fingerprint DeePHF descriptor state of type "
        f"{type(value).__name__}"
    )


class DeePHF:
    """Evaluate a perturbative correction without modifying the RHF reference."""

    @staticmethod
    def _validate_reference_object(reference):
        return validate_reference(reference)

    @staticmethod
    def _reference_state_fingerprint(reference) -> str:
        return reference_fingerprint(reference)

    def _descriptor_rank_bound(self) -> int:
        return int(np.count_nonzero(self.reference.mo_occ > 0))

    def __init__(
        self,
        reference,
        model,
        projector_basis=None,
        device="cpu",
        response_options=None,
        adjoint_options=None,
    ):
        self.reference = self._validate_reference_object(reference)
        self.device = device or "cpu"
        if isinstance(model, str):
            model = CorrNet.load(model).double()
        if isinstance(model, torch.nn.Module):
            try:
                model = model.to(self.device).eval()
            except Exception as error:
                raise DeePHFCapabilityError(
                    f"the correction model could not use device {self.device!r}: {error}"
                ) from error
        self.model = model
        if projector_basis is None:
            projector_basis = getattr(model, "_pbas", None)
        self._descriptor = AtomicDensityDescriptor(
            self.reference.mol,
            projector_basis,
        )
        self._bound_reference = self.reference
        self._bound_molecule = self.reference.mol
        self._bound_descriptor = self._descriptor
        self._bound_projector_molecule = self._descriptor.projector_mol
        self._bound_science_state_fingerprint = (
            self._current_science_state_fingerprint()
        )
        self.response_options = dict(response_options or {})
        if adjoint_options is None:
            adjoint_options = {}
        elif not isinstance(adjoint_options, Mapping):
            raise TypeError("adjoint_options must be a mapping")
        self.adjoint_options = dict(adjoint_options)
        self._trusted_response = None
        self._trusted_response_integrity = None
        self._trusted_adjoint = None
        self._trusted_adjoint_integrity = None
        self._trusted_adjoint_sensitivity_fingerprint = None
        self._trusted_adjoint_descriptor_diagnostics = None
        self._trusted_adjoint_model_fingerprint = None
        validate_model(
            self.model,
            self._descriptor.projector_basis,
            self._descriptor.n_features,
        )
        self._validated_model_output()
        self.e_base = None
        self.e_corr = None
        self.e_tot = None

    def _descriptor_science_fingerprint(self) -> str:
        descriptor = self._descriptor
        digest = hashlib.sha256()
        values = (
            f"{type(descriptor).__module__}.{type(descriptor).__qualname__}",
            descriptor.projector_basis,
            descriptor.shell_sizes,
            int(descriptor.n_features),
            descriptor.descriptor_atom_indices,
            molecule_science_fingerprint(descriptor.projector_mol),
            descriptor.overlap_shells,
        )
        for value in values:
            _update_science_digest(digest, value)
        return digest.hexdigest()

    def _current_science_state_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            self._reference_state_fingerprint(self.reference).encode("ascii")
        )
        digest.update(self._descriptor_science_fingerprint().encode("ascii"))
        return digest.hexdigest()

    def _assert_science_state(self, boundary: str) -> str:
        identities_match = (
            self.reference is self._bound_reference
            and self.reference.mol is self._bound_molecule
            and self._descriptor is self._bound_descriptor
            and self._descriptor.mol is self._bound_molecule
            and self._descriptor.projector_mol is self._bound_projector_molecule
        )
        if not identities_match:
            raise DeePHFCapabilityError(
                f"the DeePHF reference or descriptor identity changed during {boundary}"
            )
        try:
            fingerprint = self._current_science_state_fingerprint()
        except Exception as error:
            if isinstance(error, DeePHFCapabilityError):
                raise
            raise DeePHFCapabilityError(
                f"the DeePHF scientific state could not be checked during {boundary}: {error}"
            ) from error
        if fingerprint != self._bound_science_state_fingerprint:
            raise DeePHFCapabilityError(
                f"the DeePHF scientific state changed during {boundary}"
            )
        return fingerprint

    def _validate_science_state(self, boundary: str) -> str:
        self._assert_science_state(boundary)
        self._validate_reference_object(self.reference)
        return self._assert_science_state(boundary)

    def _clear_trusted_adjoint(self) -> None:
        self._trusted_adjoint = None
        self._trusted_adjoint_integrity = None
        self._trusted_adjoint_sensitivity_fingerprint = None
        self._trusted_adjoint_descriptor_diagnostics = None
        self._trusted_adjoint_model_fingerprint = None

    def _assert_force_model_state(self, fingerprint: str, boundary: str) -> None:
        validate_force_model(self.model)
        if force_model_fingerprint(self.model) != fingerprint:
            raise DeePHFCapabilityError(
                f"the force correction model state changed during {boundary}"
            )

    def _assert_trusted_force_model_state(self, boundary: str) -> None:
        if type(self._trusted_adjoint_model_fingerprint) is not str:
            raise RHFAdjointError(
                "the Z-vector force-model provenance is unavailable"
            )
        self._assert_force_model_state(
            self._trusted_adjoint_model_fingerprint,
            boundary,
        )

    @property
    def mol(self):
        return self.reference.mol

    @property
    def n_descriptor_atoms(self):
        return self._descriptor.n_descriptor_atoms

    @property
    def n_descriptor_features(self):
        return self._descriptor.n_features

    def ao_density(self):
        self._assert_science_state("AO density evaluation")
        density = spin_summed_ao_density(self.reference.make_rdm1())
        self._assert_science_state("AO density evaluation")
        return density

    def projected_density(self, flatten=False):
        return self._descriptor.projected_density(
            self.ao_density(),
            flatten=flatten,
        )

    def descriptor(self):
        return self._descriptor.descriptor(self.ao_density())

    def dq_dP(self):
        with torch.enable_grad():
            return self._descriptor.dq_dP(self.ao_density())

    def dq_dR_explicit(self):
        with torch.enable_grad():
            return self._descriptor.dq_dR_explicit(self.ao_density())

    def _descriptor_values_tensor(self) -> torch.Tensor:
        self._assert_science_state("descriptor evaluation")
        values = self._descriptor.torch_descriptor(self.ao_density())
        self._assert_science_state("descriptor evaluation")
        return values

    def _validated_model_output(self) -> torch.Tensor:
        self._assert_science_state("model evaluation")
        validate_model(
            self.model,
            self._descriptor.projector_basis,
            self._descriptor.n_features,
        )
        output = validate_model_output(
            self.model,
            self._descriptor_values_tensor(),
        )
        self._assert_science_state("model evaluation")
        return output

    def correction_sensitivity(self) -> np.ndarray:
        """Return the validated derivative of the scalar correction with respect to q."""
        self._assert_science_state("model sensitivity evaluation")
        values = self._descriptor_values_tensor()
        validate_force_model(self.model)
        validate_model(
            self.model,
            self._descriptor.projector_basis,
            self._descriptor.n_features,
        )
        model_fingerprint = force_model_fingerprint(self.model)
        if self.model is None:
            sensitivity = torch.zeros_like(values)
        else:
            evaluations = [
                self._force_energy_and_sensitivity(
                    values,
                    model_fingerprint,
                )
                for _ in range(2)
            ]
            first_energy, sensitivity = evaluations[0]
            second_energy, second_sensitivity = evaluations[1]
            if not torch.equal(first_energy, second_energy):
                raise DeePHFCapabilityError(
                    "the force correction model energy is not deterministic for "
                    "one descriptor input"
                )
            if not torch.equal(sensitivity, second_sensitivity):
                raise DeePHFCapabilityError(
                    "the force correction model descriptor sensitivity is not "
                    "deterministic for one descriptor input"
                )
            finite_difference_sensitivity = (
                self._force_descriptor_finite_difference(
                    values,
                    model_fingerprint,
                )
            )
            difference = torch.abs(
                sensitivity - finite_difference_sensitivity
            )
            tolerance = (
                _FORCE_DESCRIPTOR_FD_ATOL
                + _FORCE_DESCRIPTOR_FD_RTOL
                * torch.abs(finite_difference_sensitivity)
            )
            if torch.any(difference > tolerance):
                maximum_residual = float(torch.max(difference).cpu())
                maximum_tolerance = float(torch.max(tolerance).cpu())
                raise DeePHFCapabilityError(
                    "the force correction model descriptor sensitivity does not "
                    "match deterministic central finite differences: "
                    f"residual {maximum_residual:.3e} > "
                    f"tolerance {maximum_tolerance:.3e}"
                )
        if not torch.isfinite(sensitivity).all():
            raise DeePHFCapabilityError(
                "the correction model descriptor sensitivity must be finite"
            )
        self._assert_science_state("model sensitivity evaluation")
        return sensitivity.detach().cpu().numpy()

    def _force_energy_and_sensitivity(
        self,
        values: torch.Tensor,
        model_fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        evaluation_values = values.detach().clone().requires_grad_(True)
        rng_before = force_rng_fingerprints()
        try:
            with torch.enable_grad():
                energy = validate_model_output(self.model, evaluation_values)
                if energy.requires_grad:
                    (sensitivity,) = torch.autograd.grad(
                        energy,
                        evaluation_values,
                        torch.ones_like(energy),
                        allow_unused=True,
                    )
                else:
                    sensitivity = None
                element_constant = self._validated_element_constant()
                complete_energy = energy + energy.new_tensor(element_constant)
                if not torch.isfinite(complete_energy).all():
                    raise DeePHFCapabilityError(
                        "the complete correction energy must be finite"
                    )
        except Exception:
            self._assert_force_rng_state(rng_before)
            raise
        self._assert_force_rng_state(rng_before)
        self._assert_force_model_state(
            model_fingerprint,
            "force model energy and sensitivity evaluation",
        )
        self._assert_science_state(
            "force model energy and sensitivity evaluation"
        )
        if sensitivity is None:
            sensitivity = torch.zeros_like(evaluation_values)
        if sensitivity.shape != evaluation_values.shape:
            raise DeePHFCapabilityError(
                "the correction model descriptor sensitivity shape is invalid"
            )
        if sensitivity.dtype != torch.float64 or sensitivity.is_complex():
            raise DeePHFCapabilityError(
                "the correction model descriptor sensitivity must use real torch.float64"
            )
        if not torch.isfinite(sensitivity).all():
            raise DeePHFCapabilityError(
                "the correction model descriptor sensitivity must be finite"
            )
        ordinary_energy = self._force_ordinary_energy_invariant(
            values.detach(),
            model_fingerprint,
        )
        if not torch.equal(complete_energy.detach(), ordinary_energy):
            raise DeePHFCapabilityError(
                "the differentiable force-model energy does not match the "
                "ordinary energy for one descriptor input"
            )
        return complete_energy.detach().clone(), sensitivity.detach().clone()

    def _force_complete_energy(
        self,
        values: torch.Tensor,
        model_fingerprint: str,
    ) -> torch.Tensor:
        ordinary_values = values.detach().clone()
        rng_before = force_rng_fingerprints()
        try:
            energy = validate_model_output(self.model, ordinary_values)
            element_constant = self._validated_element_constant()
            complete_energy = energy + energy.new_tensor(element_constant)
            if not torch.isfinite(complete_energy).all():
                raise DeePHFCapabilityError(
                    "the complete correction energy must be finite"
                )
        except Exception:
            self._assert_force_rng_state(rng_before)
            raise
        self._assert_force_rng_state(rng_before)
        self._assert_force_model_state(
            model_fingerprint,
            "ordinary force-model energy evaluation",
        )
        self._assert_science_state("ordinary force-model energy evaluation")
        return complete_energy.detach().clone()

    def _force_ordinary_energy_invariant(
        self,
        values: torch.Tensor,
        model_fingerprint: str,
    ) -> torch.Tensor:
        with torch.enable_grad():
            grad_enabled_energy = self._force_complete_energy(
                values,
                model_fingerprint,
            )
        with torch.no_grad():
            no_grad_energy = self._force_complete_energy(
                values,
                model_fingerprint,
            )
        if not torch.equal(grad_enabled_energy, no_grad_energy):
            raise DeePHFCapabilityError(
                "the ordinary force-model energy depends on the Torch grad mode"
            )
        return grad_enabled_energy

    def _deterministic_force_complete_energy(
        self,
        values: torch.Tensor,
        model_fingerprint: str,
    ) -> torch.Tensor:
        first = self._force_ordinary_energy_invariant(
            values,
            model_fingerprint,
        )
        second = self._force_ordinary_energy_invariant(
            values,
            model_fingerprint,
        )
        if not torch.equal(first, second):
            raise DeePHFCapabilityError(
                "the ordinary force-model energy is not deterministic for one "
                "descriptor input"
            )
        return first

    def _force_descriptor_finite_difference(
        self,
        values: torch.Tensor,
        model_fingerprint: str,
    ) -> torch.Tensor:
        values = values.detach().clone()
        finite_difference = torch.empty_like(values)
        flat_values = values.reshape(-1)
        flat_result = finite_difference.reshape(-1)
        for index in range(flat_values.numel()):
            magnitude = abs(float(flat_values[index].detach().cpu()))
            step = _FORCE_DESCRIPTOR_FD_STEP * max(1.0, magnitude)
            forward_values = values.clone()
            backward_values = values.clone()
            forward_values.reshape(-1)[index] += step
            backward_values.reshape(-1)[index] -= step
            forward_energy = self._deterministic_force_complete_energy(
                forward_values,
                model_fingerprint,
            )
            backward_energy = self._deterministic_force_complete_energy(
                backward_values,
                model_fingerprint,
            )
            flat_result[index] = (
                float(forward_energy.detach().cpu().item())
                - float(backward_energy.detach().cpu().item())
            ) / (2.0 * step)
        if not torch.isfinite(finite_difference).all():
            raise DeePHFCapabilityError(
                "the force correction model descriptor finite difference must be finite"
            )
        return finite_difference

    @staticmethod
    def _assert_force_rng_state(before) -> None:
        after = force_rng_fingerprints()
        changed = [name for name in before if before[name] != after.get(name)]
        changed.extend(name for name in after if name not in before)
        if changed:
            raise DeePHFCapabilityError(
                "the force correction model consumed global RNG state: "
                + ", ".join(changed)
            )

    def _correction_ao_potential(self, sensitivity: np.ndarray) -> np.ndarray:
        """Contract one validated model sensitivity with dq/dP."""
        self._assert_science_state("correction AO potential construction")
        sensitivity = np.asarray(sensitivity)
        expected_shape = (
            self.n_descriptor_atoms,
            self.n_descriptor_features,
        )
        if sensitivity.shape != expected_shape:
            raise DeePHFCapabilityError(
                "the correction sensitivity shape does not match the descriptor"
            )
        if sensitivity.dtype != np.dtype(np.float64) or np.iscomplexobj(
            sensitivity
        ):
            raise DeePHFCapabilityError(
                "the correction sensitivity must be a real numpy.float64 array"
            )
        if not np.isfinite(sensitivity).all():
            raise DeePHFCapabilityError(
                "the correction sensitivity must be finite"
            )
        potential = np.einsum(
            "ap,apij->ij",
            sensitivity,
            self.dq_dP(),
        )
        if potential.shape != (self.mol.nao, self.mol.nao):
            raise DeePHFCapabilityError(
                "the correction AO potential has an invalid shape"
            )
        if potential.dtype != np.dtype(np.float64) or np.iscomplexobj(potential):
            raise DeePHFCapabilityError(
                "the correction AO potential must be a real numpy.float64 array"
            )
        if not np.isfinite(potential).all():
            raise DeePHFCapabilityError(
                "the correction AO potential must be finite"
            )
        self._assert_science_state("correction AO potential construction")
        return potential

    def correction_ao_potential(self) -> np.ndarray:
        """Return the complete model derivative d(e_corr)/dP in the AO basis."""
        return self._correction_ao_potential(self.correction_sensitivity())

    def correction_energy(self):
        if self.model is None:
            return 0.0
        tensor_energy = self._validated_model_output()
        energy = (
            float(tensor_energy.detach().cpu().item())
            + self._validated_element_constant()
        )
        if not np.isfinite(energy):
            raise DeePHFCapabilityError(
                "the complete correction energy must be finite"
            )
        return energy

    def _validated_element_constant(self) -> float:
        if self.model is None:
            return 0.0
        get_element_constant = getattr(self.model, "get_elem_const", None)
        if get_element_constant is None:
            return 0.0
        element_constant = get_element_constant(
            [int(charge) for charge in self.mol.atom_charges()]
        )
        element_constant = np.asarray(element_constant)
        if element_constant.shape != ():
            raise DeePHFCapabilityError(
                "the correction model element constant must be scalar"
            )
        if (
            element_constant.dtype != np.dtype(np.float64)
            or np.iscomplexobj(element_constant)
        ):
            raise DeePHFCapabilityError(
                "the correction model element constant must use real numpy.float64"
            )
        try:
            finite = bool(np.isfinite(element_constant).item())
        except (TypeError, ValueError) as error:
            raise DeePHFCapabilityError(
                "the correction model element constant must use real numpy.float64"
            ) from error
        if not finite:
            raise DeePHFCapabilityError(
                "the correction model element constant must be real and finite"
            )
        return float(element_constant.item())

    def validate_force_compatibility(self, **tolerances):
        """Validate ordered-eigenvalue and model-sensitivity force semantics."""
        validate_force_model(self.model)
        self._validate_reference_object(self.reference)
        sensitivity = self.correction_sensitivity()
        validate_force_model(self.model)
        return self._validate_force_compatibility_with_sensitivity(
            sensitivity,
            **tolerances,
        )

    def _validate_force_compatibility_with_sensitivity(
        self,
        sensitivity,
        **tolerances,
    ):
        validate_force_model(self.model)
        values = self._descriptor_values_tensor()
        n_occupied = self._descriptor_rank_bound()
        diagnostics = validate_differentiability(
            values.detach().cpu().numpy(),
            self._descriptor.shell_sizes,
            n_occupied,
            sensitivity,
            **tolerances,
        )
        validate_force_model(self.model)
        return diagnostics

    def response(self, **response_options) -> RHFResponse:
        """Solve the audited complete first-order RHF density response."""
        self.validate_force_compatibility()
        options = _validated_backend_options(
            self.response_options,
            response_options,
            _DIRECT_RESPONSE_OPTIONS,
            "direct",
        )
        response = RHFResponseAdapter(self.reference, **options).solve()
        self._trusted_response = response
        self._trusted_response_integrity = response.integrity_fingerprint
        return response

    def adjoint(self, **adjoint_options) -> RHFAdjoint:
        """Solve one audited correction-specific RHF scalar adjoint."""
        inputs = self._zvector_inputs(
            adjoint_options
        )
        return self._validate_zvector_inputs(*inputs)[2]

    def _zvector_inputs(self, adjoint_options):
        """Evaluate one internally consistent sensitivity and scalar adjoint."""
        self._clear_trusted_adjoint()
        self._assert_science_state("Z-vector input evaluation")
        validate_force_model(self.model)
        model_fingerprint = force_model_fingerprint(self.model)
        self._validate_reference_object(self.reference)
        self._assert_science_state("Z-vector reference validation")
        sensitivity = _immutable_array(self.correction_sensitivity())
        self._assert_force_model_state(
            model_fingerprint,
            "Z-vector model sensitivity evaluation",
        )
        self._validate_science_state("Z-vector model sensitivity evaluation")
        descriptor_diagnostics = (
            self._validate_force_compatibility_with_sensitivity(sensitivity)
        )
        self._assert_science_state("Z-vector descriptor validation")
        self._assert_force_model_state(
            model_fingerprint,
            "Z-vector descriptor validation",
        )
        options = _validated_backend_options(
            self.adjoint_options,
            adjoint_options,
            _ZVECTOR_OPTIONS,
            "zvector",
        )
        objective_ao_potential = self._correction_ao_potential(sensitivity)
        self._validate_science_state("Z-vector AO potential construction")
        self._assert_force_model_state(
            model_fingerprint,
            "Z-vector AO potential construction",
        )
        adjoint = RHFAdjointAdapter(self.reference, **options).solve(
            objective_ao_potential
        )
        self._assert_science_state("Z-vector adjoint construction")
        self._assert_force_model_state(
            model_fingerprint,
            "Z-vector adjoint construction",
        )
        self._trusted_adjoint = adjoint
        self._trusted_adjoint_integrity = adjoint.integrity_fingerprint
        self._trusted_adjoint_sensitivity_fingerprint = _array_fingerprint(
            sensitivity
        )
        self._trusted_adjoint_descriptor_diagnostics = descriptor_diagnostics
        self._trusted_adjoint_model_fingerprint = model_fingerprint
        return descriptor_diagnostics, sensitivity, adjoint

    def _validate_zvector_inputs(
        self,
        descriptor_diagnostics,
        sensitivity,
        adjoint,
    ):
        """Audit one Z-vector tuple before any gradient data are consumed."""
        self._assert_science_state("Z-vector result consumption")
        if type(descriptor_diagnostics) is not DescriptorDiagnostics:
            raise RHFAdjointError(
                "the supplied descriptor diagnostics have an invalid type"
            )
        if type(adjoint) is not RHFAdjoint:
            raise RHFAdjointError("the supplied RHF adjoint has an invalid type")
        if type(adjoint.diagnostics) is not RHFAdjointDiagnostics:
            raise RHFAdjointError(
                "the supplied RHF adjoint diagnostics have an invalid type"
            )
        if adjoint is not self._trusted_adjoint:
            raise RHFAdjointError(
                "the supplied RHF adjoint was not produced by this DeePHF evaluation"
            )
        self._assert_trusted_force_model_state("Z-vector result consumption")
        if (
            adjoint.integrity_fingerprint
            != self._trusted_adjoint_integrity
            or adjoint.integrity_fingerprint
            != adjoint_integrity_fingerprint(adjoint)
        ):
            raise RHFAdjointError(
                "the supplied RHF adjoint failed its integrity check"
            )
        if not isinstance(sensitivity, np.ndarray):
            raise RHFAdjointError(
                "the supplied correction sensitivity has an invalid type"
            )
        expected_sensitivity_shape = (
            self.n_descriptor_atoms,
            self.n_descriptor_features,
        )
        if sensitivity.shape != expected_sensitivity_shape:
            raise RHFAdjointError(
                "the supplied correction sensitivity has an invalid shape"
            )
        if sensitivity.dtype != np.dtype(np.float64) or np.iscomplexobj(
            sensitivity
        ):
            raise RHFAdjointError(
                "the supplied correction sensitivity must use real numpy.float64"
            )
        if not np.isfinite(sensitivity).all() or sensitivity.flags.writeable:
            raise RHFAdjointError(
                "the supplied correction sensitivity must be finite and immutable"
            )
        if (
            _array_fingerprint(sensitivity)
            != self._trusted_adjoint_sensitivity_fingerprint
        ):
            raise RHFAdjointError(
                "the supplied correction sensitivity failed its integrity check"
            )
        if descriptor_diagnostics != self._trusted_adjoint_descriptor_diagnostics:
            raise RHFAdjointError(
                "the supplied descriptor diagnostics do not match this evaluation"
            )
        expected_objective_ao_potential = self._correction_ao_potential(
            sensitivity
        )
        diagnostics = adjoint.diagnostics
        audit_adapter = RHFAdjointAdapter(
            self.reference,
            residual_tolerance=diagnostics.residual_tolerance,
            orbital_gap_tolerance=diagnostics.orbital_gap_tolerance,
            operator_stability_tolerance=(
                diagnostics.operator_stability_tolerance
            ),
            operator_condition_tolerance=(
                diagnostics.operator_condition_tolerance
            ),
            operator_symmetry_tolerance=(
                diagnostics.operator_symmetry_tolerance
            ),
            operator_dimension_limit=diagnostics.operator_dimension_limit,
            objective_symmetry_tolerance=(
                diagnostics.objective_symmetry_tolerance
            ),
        )
        audit_adapter.audit_adjoint(
            adjoint,
            expected_objective_ao_potential,
        )
        self._assert_science_state("Z-vector result audit")
        self._assert_trusted_force_model_state("Z-vector result audit")
        return descriptor_diagnostics, sensitivity, adjoint

    def _validate_response(self, response: RHFResponse) -> RHFResponse:
        """Reject response data that are stale, foreign, mutable, or incomplete."""
        self._validate_reference_object(self.reference)
        if type(response) is not RHFResponse:
            raise RHFResponseError("the supplied RHF response has an invalid type")
        if type(response.diagnostics) is not RHFResponseDiagnostics:
            raise RHFResponseError(
                "the supplied RHF response diagnostics have an invalid type"
            )
        if response.reference_identity != id(self.reference):
            raise RHFResponseError("the supplied RHF response belongs to another reference")
        if response.state_fingerprint != reference_fingerprint(self.reference):
            raise RHFResponseError("the supplied RHF response does not match the current RHF state")
        if response.integrity_fingerprint != response_integrity_fingerprint(response):
            raise RHFResponseError("the supplied RHF response failed its integrity check")
        natm = self.mol.natm
        nao = self.mol.nao
        nmo = self.reference.mo_coeff.shape[1]
        nocc = int(np.count_nonzero(self.reference.mo_occ > 0))
        nvir = nmo - nocc
        expected_shapes = {
            "mo_response": (natm, 3, nmo, nocc),
            "mo_response_occupied_virtual": (natm, 3, nmo, nocc),
            "mo_response_metric": (natm, 3, nmo, nocc),
            "coefficient_response": (natm, 3, nao, nocc),
            "coefficient_response_occupied_virtual": (natm, 3, nao, nocc),
            "coefficient_response_metric": (natm, 3, nao, nocc),
            "density_response": (natm, 3, nao, nao),
            "density_response_occupied_virtual": (natm, 3, nao, nao),
            "density_response_metric": (natm, 3, nao, nao),
            "overlap_derivative": (natm, 3, nao, nao),
            "hamiltonian_derivative": (natm, 3, nao, nao),
            "orbital_response_residual": (natm, 3, nvir, nocc),
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(response, name)
            if not isinstance(value, np.ndarray) or value.shape != expected_shape:
                raise RHFResponseError(
                    f"the supplied RHF response field {name} has shape "
                    f"{getattr(value, 'shape', None)}; expected {expected_shape}"
                )
            if np.iscomplexobj(value):
                raise RHFResponseError(
                    f"the supplied RHF response field {name} must be real"
                )
            if value.dtype != np.dtype(np.float64):
                raise RHFResponseError(
                    f"the supplied RHF response field {name} must use numpy.float64"
                )
            if not np.isfinite(value).all():
                raise RHFResponseError(
                    f"the supplied RHF response field {name} must be finite"
                )
            if value.flags.writeable:
                raise RHFResponseError(
                    f"the supplied RHF response field {name} must be immutable"
                )
        component_pairs = (
            (
                response.mo_response,
                response.mo_response_occupied_virtual,
                response.mo_response_metric,
                "MO response",
            ),
            (
                response.coefficient_response,
                response.coefficient_response_occupied_virtual,
                response.coefficient_response_metric,
                "coefficient response",
            ),
            (
                response.density_response,
                response.density_response_occupied_virtual,
                response.density_response_metric,
                "density response",
            ),
        )
        for total, occupied_virtual, metric, label in component_pairs:
            if not np.allclose(total, occupied_virtual + metric, rtol=0.0, atol=1.0e-12):
                raise RHFResponseError(
                    f"the supplied RHF {label} components are inconsistent"
                )
        coefficient = np.asarray(self.reference.mo_coeff)
        occupations = np.asarray(self.reference.mo_occ)
        occupied = occupations > 0
        virtual = occupations == 0
        occupied_coefficients = coefficient[:, occupied]
        if np.max(
            np.abs(response.mo_response_occupied_virtual[..., occupied, :]),
            initial=0.0,
        ) > 1.0e-12:
            raise RHFResponseError(
                "the supplied occupied-virtual response has occupied-space support"
            )
        if np.max(
            np.abs(response.mo_response_metric[..., virtual, :]),
            initial=0.0,
        ) > 1.0e-12:
            raise RHFResponseError(
                "the supplied metric response has virtual-space support"
            )
        response_levels = (
            (
                response.mo_response,
                response.coefficient_response,
                response.density_response,
                "complete",
            ),
            (
                response.mo_response_occupied_virtual,
                response.coefficient_response_occupied_virtual,
                response.density_response_occupied_virtual,
                "occupied-virtual",
            ),
            (
                response.mo_response_metric,
                response.coefficient_response_metric,
                response.density_response_metric,
                "metric",
            ),
        )
        for mo_response, coefficient_response, density_response, label in response_levels:
            expected_coefficient_response = np.einsum(
                "mp,...pi->...mi",
                coefficient,
                mo_response,
            )
            expected_density_response = np.einsum(
                "...pi,qi,i->...pq",
                expected_coefficient_response,
                occupied_coefficients,
                occupations[occupied],
            )
            expected_density_response = (
                expected_density_response
                + expected_density_response.swapaxes(-1, -2)
            )
            if not np.allclose(
                coefficient_response,
                expected_coefficient_response,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RHFResponseError(
                    f"the supplied RHF {label} coefficient response is inconsistent"
                )
            if not np.allclose(
                density_response,
                expected_density_response,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RHFResponseError(
                    f"the supplied RHF {label} density response is inconsistent"
                )
        diagnostics = response.diagnostics
        diagnostic_values = (
            diagnostics.minimum_orbital_gap,
            diagnostics.cphf_tolerance,
            diagnostics.maximum_residual,
            diagnostics.residual_rms,
            diagnostics.residual_tolerance,
            diagnostics.invariant_tolerance,
            diagnostics.orbital_gap_tolerance,
            diagnostics.level_shift,
            diagnostics.operator_stability_tolerance,
            diagnostics.operator_condition_tolerance,
            diagnostics.operator_symmetry_tolerance,
            diagnostics.operator_minimum_eigenvalue,
            diagnostics.operator_maximum_eigenvalue,
            diagnostics.operator_condition_number,
            diagnostics.operator_symmetry_residual,
            diagnostics.metric_residual,
            diagnostics.idempotency_residual,
            diagnostics.particle_number_residual,
            *diagnostics.residual_history,
        )
        if not np.isfinite(diagnostic_values).all():
            raise RHFResponseError("the supplied RHF response diagnostics must be finite")
        if (
            diagnostics.cphf_tolerance <= 0
            or diagnostics.residual_tolerance <= 0
            or diagnostics.invariant_tolerance <= 0
            or diagnostics.orbital_gap_tolerance <= 0
            or diagnostics.max_cycle <= 0
            or diagnostics.max_refinement_cycles < 0
            or diagnostics.operator_stability_tolerance <= 0
            or diagnostics.operator_condition_tolerance <= 1
            or diagnostics.operator_symmetry_tolerance <= 0
            or diagnostics.operator_dimension_limit <= 0
            or diagnostics.response_dimension <= 0
            or diagnostics.response_dimension != nocc * nvir
            or diagnostics.response_dimension > diagnostics.operator_dimension_limit
            or diagnostics.operator_minimum_eigenvalue
            <= diagnostics.operator_stability_tolerance
            or diagnostics.operator_condition_number
            > diagnostics.operator_condition_tolerance
            or diagnostics.operator_symmetry_residual
            > diagnostics.operator_symmetry_tolerance
        ):
            raise RHFResponseError("the supplied RHF response controls are invalid")
        measured_residual = float(
            np.max(np.abs(response.orbital_response_residual), initial=0.0)
        )
        measured_residual_rms = float(
            np.sqrt(np.mean(np.square(response.orbital_response_residual)))
        )
        if not np.isclose(
            diagnostics.maximum_residual,
            measured_residual,
            rtol=1.0e-12,
            atol=np.finfo(float).eps,
        ):
            raise RHFResponseError("the supplied RHF response residual diagnostic is inconsistent")
        if not np.isclose(
            diagnostics.residual_rms,
            measured_residual_rms,
            rtol=1.0e-12,
            atol=np.finfo(float).eps,
        ):
            raise RHFResponseError("the supplied RHF response RMS residual is inconsistent")
        if (
            not diagnostics.residual_history
            or diagnostics.refinement_cycles != len(diagnostics.residual_history) - 1
            or diagnostics.refinement_cycles > diagnostics.max_refinement_cycles
            or not np.isclose(
                diagnostics.residual_history[-1],
                diagnostics.maximum_residual,
                rtol=1.0e-12,
                atol=np.finfo(float).eps,
            )
        ):
            raise RHFResponseError("the supplied RHF response residual history is inconsistent")
        if diagnostics.maximum_residual > diagnostics.residual_tolerance:
            raise RHFResponseError("the supplied RHF response residual exceeds its tolerance")
        invariant_values = (
            diagnostics.metric_residual,
            diagnostics.idempotency_residual,
            diagnostics.particle_number_residual,
        )
        if max(invariant_values) > diagnostics.invariant_tolerance:
            raise RHFResponseError("the supplied RHF response invariant exceeds its tolerance")
        trusted_response_is_unchanged = (
            response is self._trusted_response
            and response.integrity_fingerprint
            == self._trusted_response_integrity
        )
        if not trusted_response_is_unchanged:
            RHFResponseAdapter(
                self.reference,
                cphf_tolerance=diagnostics.cphf_tolerance,
                residual_tolerance=diagnostics.residual_tolerance,
                invariant_tolerance=diagnostics.invariant_tolerance,
                orbital_gap_tolerance=diagnostics.orbital_gap_tolerance,
                max_cycle=diagnostics.max_cycle,
                max_refinement_cycles=diagnostics.max_refinement_cycles,
                level_shift=diagnostics.level_shift,
                operator_stability_tolerance=(
                    diagnostics.operator_stability_tolerance
                ),
                operator_condition_tolerance=(
                    diagnostics.operator_condition_tolerance
                ),
                operator_symmetry_tolerance=(
                    diagnostics.operator_symmetry_tolerance
                ),
                operator_dimension_limit=diagnostics.operator_dimension_limit,
            ).audit_response_equations(response)
        return response

    def first_order_density(
        self,
        response: RHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return the complete numerical AO density derivative dP/dR."""
        if response is not None and response_options:
            raise ValueError(
                "response and response_options are mutually exclusive"
            )
        if response is None:
            response = self.response(**response_options)
        return self._validate_response(response).density_response

    def dq_dR_response(
        self,
        response: RHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return the descriptor derivative generated by the RHF density response."""
        self.validate_force_compatibility()
        density_response = self.first_order_density(
            response=response,
            **response_options,
        )
        return np.einsum(
            "apij,bxij->bxap",
            self.dq_dP(),
            density_response,
        )

    def dq_dR_relaxed(
        self,
        response: RHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return dq/dR including explicit projector motion and RHF response."""
        if response is None:
            response = self.response(**response_options)
            response_options = {}
        return self.dq_dR_explicit() + self.dq_dR_response(
            response=response,
            **response_options,
        )

    def nuc_grad_method(self, *, backend="direct", **backend_options):
        """Build one explicitly selected strict analytic-gradient backend."""
        if type(backend) is not str or backend not in {"direct", "zvector"}:
            raise ValueError("gradient backend must be 'direct' or 'zvector'")
        if backend == "direct":
            from .gradient import RHFDeePHFGradients

            _validated_backend_options(
                self.response_options,
                backend_options,
                _DIRECT_RESPONSE_OPTIONS,
                backend,
            )
            return RHFDeePHFGradients(
                self,
                response_options=backend_options,
            )
        from .zvector import RHFDeePHFZVectorGradients

        _validated_backend_options(
            self.adjoint_options,
            backend_options,
            _ZVECTOR_OPTIONS,
            backend,
        )
        return RHFDeePHFZVectorGradients(
            self,
            adjoint_options=backend_options,
        )

    def gradient(self, *, backend="direct", **backend_options) -> np.ndarray:
        """Evaluate the strict analytic nuclear energy gradient."""
        return self.nuc_grad_method(
            backend=backend,
            **backend_options,
        ).kernel()

    def forces(self, *, backend="direct", **backend_options) -> np.ndarray:
        """Evaluate nuclear forces as the negative analytic gradient."""
        return -self.gradient(backend=backend, **backend_options)

    def kernel(self):
        """Evaluate E_base + E_corr while leaving the reference unchanged."""
        self._validate_reference_object(self.reference)
        self.e_base = float(self.reference.e_tot)
        self.e_corr = self.correction_energy()
        self.e_tot = self.e_base + self.e_corr
        if not np.isfinite(self.e_tot):
            raise DeePHFCapabilityError("the total DeePHF energy must be finite")
        return self.e_tot
