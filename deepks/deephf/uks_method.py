"""Perturbative DeePHF energy method for a strict finite-grid UKS reference."""

import numpy as np

from deepks.descriptor import DescriptorDiagnostics

from .capabilities import force_model_fingerprint, validate_force_model
from .method import (
    _DIRECT_RESPONSE_OPTIONS,
    _array_fingerprint,
    _immutable_array,
    _validated_backend_options,
)
from .pyscf_uks import (
    UKSAdjoint,
    UKSAdjointAdapter,
    UKSAdjointDiagnostics,
    UKSAdjointError,
    UKSResponse,
    UKSResponseAdapter,
    UKSResponseDiagnostics,
    UKSResponseError,
    uks_adjoint_integrity_fingerprint,
    uks_reference_fingerprint,
    validate_uks_reference,
)
from .uhf_method import UHFDeePHF


_TRUSTED_RESPONSE_LIMIT = 8
_UKS_ZVECTOR_OPTIONS = frozenset(
    {
        "residual_tolerance",
        "invariant_tolerance",
        "orbital_gap_tolerance",
        "operator_stability_tolerance",
        "operator_condition_tolerance",
        "operator_symmetry_tolerance",
        "operator_dimension_limit",
        "objective_symmetry_tolerance",
        "max_cycle",
        "krylov_restart",
    }
)


class UKSDeePHF(UHFDeePHF):
    """Evaluate a perturbative correction around one strict UKS reference."""

    @staticmethod
    def _validate_reference_object(reference):
        return validate_uks_reference(reference)

    @staticmethod
    def _reference_state_fingerprint(reference) -> str:
        return uks_reference_fingerprint(reference)

    def __init__(
        self,
        reference,
        model,
        projector_basis=None,
        device="cpu",
        response_options=None,
        adjoint_options=None,
    ):
        super().__init__(
            reference,
            model,
            projector_basis=projector_basis,
            device=device,
            response_options=response_options,
            adjoint_options=adjoint_options,
        )
        self._trusted_uks_responses = {}
        self._trusted_uks_adjoint_adapter = None
        self._trusted_uks_adjoint_controls = None

    def _clear_trusted_adjoint(self) -> None:
        super()._clear_trusted_adjoint()
        self._trusted_uks_adjoint_adapter = None
        self._trusted_uks_adjoint_controls = None

    @staticmethod
    def _uks_adjoint_controls(adapter) -> tuple:
        return tuple((name, getattr(adapter, name)) for name in sorted(_UKS_ZVECTOR_OPTIONS))

    def _assert_trusted_uks_force_model_state(self, boundary: str) -> None:
        fingerprint = self._trusted_adjoint_model_fingerprint
        if type(fingerprint) is not str:
            raise UKSAdjointError("the UKS Z-vector force-model provenance is unavailable")
        self._assert_force_model_state(fingerprint, boundary)

    def response(self, **response_options) -> UKSResponse:
        """Solve the audited complete finite-grid UKS density response."""
        self._trusted_response = None
        self._trusted_response_integrity = None
        self.validate_force_compatibility()
        options = _validated_backend_options(
            self.response_options,
            response_options,
            _DIRECT_RESPONSE_OPTIONS,
            "direct",
        )
        adapter = UKSResponseAdapter(self.reference, **options)
        response = adapter.solve()
        self._trusted_response = response
        self._trusted_response_integrity = response.integrity_fingerprint
        self._trusted_uks_responses[id(response)] = (
            response,
            adapter,
            response.integrity_fingerprint,
        )
        while len(self._trusted_uks_responses) > _TRUSTED_RESPONSE_LIMIT:
            self._trusted_uks_responses.pop(next(iter(self._trusted_uks_responses)))
        return response

    def _validate_response(self, response: UKSResponse) -> UKSResponse:
        """Audit one response produced by this exact UKS method."""
        self._validate_reference_object(self.reference)
        if type(response) is not UKSResponse or type(response.diagnostics) is not UKSResponseDiagnostics:
            raise UKSResponseError("the supplied UKS response has an invalid type")
        trusted = self._trusted_uks_responses.get(id(response))
        if trusted is None or trusted[0] is not response:
            raise UKSResponseError("the supplied UKS response was not produced by this UKS DeePHF method")
        _, adapter, original_integrity = trusted
        if response.integrity_fingerprint != original_integrity:
            raise UKSResponseError("the supplied UKS response changed after it was produced")
        if type(adapter) is not UKSResponseAdapter:
            raise UKSResponseError("the trusted UKS response adapter is unavailable")
        adapter.audit_response_equations(response)
        return response

    def adjoint(self, **adjoint_options) -> UKSAdjoint:
        """Solve one audited correction-specific coupled UKS adjoint."""
        self._clear_trusted_adjoint()
        try:
            inputs = self._zvector_inputs(adjoint_options)
            return self._validate_zvector_inputs(*inputs)[2]
        except Exception:
            self._clear_trusted_adjoint()
            raise

    def _zvector_inputs(self, adjoint_options):
        self._clear_trusted_adjoint()
        try:
            self._assert_science_state("UKS Z-vector input evaluation")
            validate_force_model(self.model)
            model_fingerprint = force_model_fingerprint(self.model)
            self._validate_reference_object(self.reference)
            self._assert_science_state("UKS Z-vector reference validation")
            sensitivity = _immutable_array(self.correction_sensitivity())
            self._assert_force_model_state(model_fingerprint, "UKS Z-vector model sensitivity evaluation")
            self._validate_science_state("UKS Z-vector model sensitivity evaluation")
            descriptor_diagnostics = self._validate_force_compatibility_with_sensitivity(sensitivity)
            self._assert_science_state("UKS Z-vector descriptor validation")
            self._assert_force_model_state(model_fingerprint, "UKS Z-vector descriptor validation")
            options = _validated_backend_options(
                self.adjoint_options,
                adjoint_options,
                _UKS_ZVECTOR_OPTIONS,
                "zvector",
            )
            objective = self._correction_ao_potential(sensitivity)
            self._validate_science_state("UKS Z-vector AO potential construction")
            self._assert_force_model_state(model_fingerprint, "UKS Z-vector AO potential construction")
            adapter = UKSAdjointAdapter(self.reference, **options)
            adjoint = adapter.solve(objective)
            self._assert_science_state("UKS Z-vector adjoint construction")
            self._assert_force_model_state(model_fingerprint, "UKS Z-vector adjoint construction")
            self._trusted_adjoint = adjoint
            self._trusted_adjoint_integrity = adjoint.integrity_fingerprint
            self._trusted_adjoint_sensitivity_fingerprint = _array_fingerprint(sensitivity)
            self._trusted_adjoint_descriptor_diagnostics = descriptor_diagnostics
            self._trusted_adjoint_model_fingerprint = model_fingerprint
            self._trusted_uks_adjoint_adapter = adapter
            self._trusted_uks_adjoint_controls = self._uks_adjoint_controls(adapter)
            return descriptor_diagnostics, sensitivity, adjoint
        except Exception:
            self._clear_trusted_adjoint()
            raise

    def _validate_zvector_inputs(self, descriptor_diagnostics, sensitivity, adjoint):
        self._assert_science_state("UKS Z-vector result consumption")
        if type(descriptor_diagnostics) is not DescriptorDiagnostics:
            raise UKSAdjointError("the supplied descriptor diagnostics have an invalid type")
        if type(adjoint) is not UKSAdjoint or type(adjoint.diagnostics) is not UKSAdjointDiagnostics:
            raise UKSAdjointError("the supplied UKS adjoint has an invalid type")
        if adjoint is not self._trusted_adjoint:
            raise UKSAdjointError("the supplied UKS adjoint was not produced by this evaluation")
        self._assert_trusted_uks_force_model_state("UKS Z-vector result consumption")
        if (
            type(self._trusted_adjoint_integrity) is not str
            or adjoint.integrity_fingerprint != self._trusted_adjoint_integrity
            or adjoint.integrity_fingerprint != uks_adjoint_integrity_fingerprint(adjoint)
        ):
            raise UKSAdjointError("the supplied UKS adjoint failed its integrity check")
        expected_shape = (self.n_descriptor_atoms, self.n_descriptor_features)
        if type(sensitivity) is not np.ndarray or sensitivity.shape != expected_shape:
            raise UKSAdjointError("the supplied correction sensitivity has an invalid shape or type")
        if sensitivity.dtype != np.dtype(np.float64) or np.iscomplexobj(sensitivity) or not np.isfinite(sensitivity).all() or sensitivity.flags.writeable:
            raise UKSAdjointError("the supplied correction sensitivity must be immutable finite float64")
        if _array_fingerprint(sensitivity) != self._trusted_adjoint_sensitivity_fingerprint:
            raise UKSAdjointError("the supplied correction sensitivity failed its integrity check")
        if descriptor_diagnostics != self._trusted_adjoint_descriptor_diagnostics:
            raise UKSAdjointError("the supplied descriptor diagnostics do not match this evaluation")
        adapter = self._trusted_uks_adjoint_adapter
        if type(adapter) is not UKSAdjointAdapter:
            raise UKSAdjointError("the trusted UKS adjoint adapter is unavailable")
        if self._uks_adjoint_controls(adapter) != self._trusted_uks_adjoint_controls:
            raise UKSAdjointError("the trusted UKS adjoint controls changed after the solve")
        expected_objective = self._correction_ao_potential(sensitivity)
        adapter.audit_adjoint(adjoint, expected_objective)
        self._assert_science_state("UKS Z-vector result audit")
        self._assert_trusted_uks_force_model_state("UKS Z-vector result audit")
        return descriptor_diagnostics, sensitivity, adjoint

    def nuc_grad_method(self, *, backend="direct", **backend_options):
        """Build one explicitly selected finite-grid UKS gradient backend."""
        if type(backend) is not str or backend not in {"direct", "zvector"}:
            raise ValueError("UKS gradient backend must be 'direct' or 'zvector'")
        if backend == "direct":
            _validated_backend_options(self.response_options, backend_options, _DIRECT_RESPONSE_OPTIONS, backend)
            from .uks_gradient import UKSDeePHFGradients

            return UKSDeePHFGradients(self, response_options=backend_options)
        _validated_backend_options(self.adjoint_options, backend_options, _UKS_ZVECTOR_OPTIONS, backend)
        from .uks_zvector import UKSDeePHFZVectorGradients

        return UKSDeePHFZVectorGradients(self, adjoint_options=backend_options)


__all__ = ["UKSDeePHF"]
