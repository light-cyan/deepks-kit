"""Perturbative DeePHF energy method for a strict native RKS reference."""

import numpy as np

from deepks.descriptor import DescriptorDiagnostics

from .capabilities import (
    force_model_fingerprint,
    validate_force_model,
)
from .method import (
    DeePHF,
    _DIRECT_RESPONSE_OPTIONS,
    _array_fingerprint,
    _immutable_array,
    _validated_backend_options,
)
from .pyscf_rks import (
    RKSAdjoint,
    RKSAdjointAdapter,
    RKSAdjointDiagnostics,
    RKSAdjointError,
    RKSResponse,
    RKSResponseAdapter,
    RKSResponseError,
    rks_adjoint_integrity_fingerprint,
    rks_reference_fingerprint,
    validate_rks_reference,
)


_TRUSTED_RESPONSE_LIMIT = 8
_RKS_ZVECTOR_OPTIONS = frozenset(
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


class RKSDeePHF(DeePHF):
    """Evaluate a perturbative correction around one strict RKS reference."""

    @staticmethod
    def _validate_reference_object(reference):
        return validate_rks_reference(reference)

    @staticmethod
    def _reference_state_fingerprint(reference) -> str:
        return rks_reference_fingerprint(reference)

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
        self._trusted_rks_responses = {}
        self._trusted_rks_adjoint_adapter = None
        self._trusted_rks_adjoint_controls = None

    def _clear_trusted_adjoint(self) -> None:
        """Clear every RKS correction-specific adjoint provenance binding."""
        super()._clear_trusted_adjoint()
        self._trusted_rks_adjoint_adapter = None
        self._trusted_rks_adjoint_controls = None

    @staticmethod
    def _rks_adjoint_controls(adapter) -> tuple:
        return tuple(
            (name, getattr(adapter, name))
            for name in sorted(_RKS_ZVECTOR_OPTIONS)
        )

    def _assert_trusted_rks_force_model_state(self, boundary: str) -> None:
        fingerprint = self._trusted_adjoint_model_fingerprint
        if type(fingerprint) is not str:
            raise RKSAdjointError(
                "the RKS Z-vector force-model provenance is unavailable"
            )
        self._assert_force_model_state(fingerprint, boundary)

    def response(self, **response_options) -> RKSResponse:
        """Solve the audited complete finite-grid RKS density response."""
        self._trusted_response = None
        self._trusted_response_integrity = None
        self.validate_force_compatibility()
        options = _validated_backend_options(
            self.response_options,
            response_options,
            _DIRECT_RESPONSE_OPTIONS,
            "direct",
        )
        adapter = RKSResponseAdapter(self.reference, **options)
        response = adapter.solve()
        self._trusted_response = response
        self._trusted_response_integrity = response.integrity_fingerprint
        self._trusted_rks_responses[id(response)] = (
            response,
            adapter,
            response.integrity_fingerprint,
        )
        while len(self._trusted_rks_responses) > _TRUSTED_RESPONSE_LIMIT:
            self._trusted_rks_responses.pop(next(iter(self._trusted_rks_responses)))
        return response

    def _validate_response(self, response: RKSResponse) -> RKSResponse:
        """Rebuild and audit one supplied finite-grid RKS response."""
        self._validate_reference_object(self.reference)
        if type(response) is not RKSResponse:
            raise RKSResponseError("the supplied RKS response has an invalid type")
        trusted = self._trusted_rks_responses.get(id(response))
        if trusted is None or trusted[0] is not response:
            raise RKSResponseError(
                "the supplied RKS response was not produced by this RKS DeePHF method"
            )
        _, adapter, original_integrity = trusted
        if (
            type(original_integrity) is not str
            or response.integrity_fingerprint != original_integrity
        ):
            raise RKSResponseError(
                "the supplied RKS response changed after it was produced"
            )
        if type(adapter) is not RKSResponseAdapter:
            raise RKSResponseError(
                "the trusted RKS response adapter is unavailable"
            )
        adapter.audit_response_equations(response)
        return response

    def first_order_density(
        self,
        response: RKSResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return the complete numerical AO density derivative dP/dR."""
        if response is not None and response_options:
            raise ValueError("response and response_options are mutually exclusive")
        if response is None:
            response = self.response(**response_options)
        return self._validate_response(response).density_response

    def dq_dR_response(
        self,
        response: RKSResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return the descriptor derivative generated by the RKS response."""
        self.validate_force_compatibility()
        density_response = self.first_order_density(
            response=response,
            **response_options,
        )
        result = np.einsum(
            "apij,bxij->bxap",
            self.dq_dP(),
            density_response,
        )
        if not np.isfinite(result).all():
            raise RKSResponseError(
                "the RKS descriptor response is nonfinite"
            )
        return result

    def dq_dR_relaxed(
        self,
        response: RKSResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return complete relaxed dq/dR for the RKS reference."""
        if response is None:
            response = self.response(**response_options)
            response_options = {}
        return self.dq_dR_explicit() + self.dq_dR_response(
            response=response,
            **response_options,
        )

    def adjoint(self, **adjoint_options) -> RKSAdjoint:
        """Solve one audited correction-specific RKS scalar adjoint."""
        self._clear_trusted_adjoint()
        try:
            inputs = self._zvector_inputs(adjoint_options)
            return self._validate_zvector_inputs(*inputs)[2]
        except Exception:
            self._clear_trusted_adjoint()
            raise

    def _zvector_inputs(self, adjoint_options):
        """Evaluate one internally consistent RKS sensitivity and adjoint."""
        self._clear_trusted_adjoint()
        try:
            self._assert_science_state("RKS Z-vector input evaluation")
            validate_force_model(self.model)
            model_fingerprint = force_model_fingerprint(self.model)
            self._validate_reference_object(self.reference)
            self._assert_science_state("RKS Z-vector reference validation")
            sensitivity = _immutable_array(self.correction_sensitivity())
            self._assert_force_model_state(
                model_fingerprint,
                "RKS Z-vector model sensitivity evaluation",
            )
            self._validate_science_state(
                "RKS Z-vector model sensitivity evaluation"
            )
            descriptor_diagnostics = (
                self._validate_force_compatibility_with_sensitivity(sensitivity)
            )
            self._assert_science_state("RKS Z-vector descriptor validation")
            self._assert_force_model_state(
                model_fingerprint,
                "RKS Z-vector descriptor validation",
            )
            options = _validated_backend_options(
                self.adjoint_options,
                adjoint_options,
                _RKS_ZVECTOR_OPTIONS,
                "zvector",
            )
            objective_ao_potential = self._correction_ao_potential(sensitivity)
            self._validate_science_state(
                "RKS Z-vector AO potential construction"
            )
            self._assert_force_model_state(
                model_fingerprint,
                "RKS Z-vector AO potential construction",
            )
            adapter = RKSAdjointAdapter(self.reference, **options)
            adjoint = adapter.solve(objective_ao_potential)
            self._assert_science_state("RKS Z-vector adjoint construction")
            self._assert_force_model_state(
                model_fingerprint,
                "RKS Z-vector adjoint construction",
            )
            self._trusted_adjoint = adjoint
            self._trusted_adjoint_integrity = adjoint.integrity_fingerprint
            self._trusted_adjoint_sensitivity_fingerprint = _array_fingerprint(
                sensitivity
            )
            self._trusted_adjoint_descriptor_diagnostics = descriptor_diagnostics
            self._trusted_adjoint_model_fingerprint = model_fingerprint
            self._trusted_rks_adjoint_adapter = adapter
            self._trusted_rks_adjoint_controls = self._rks_adjoint_controls(
                adapter
            )
            return descriptor_diagnostics, sensitivity, adjoint
        except Exception:
            self._clear_trusted_adjoint()
            raise

    def _validate_zvector_inputs(
        self,
        descriptor_diagnostics,
        sensitivity,
        adjoint,
    ):
        """Audit one trusted RKS Z-vector tuple before consuming gradients."""
        self._assert_science_state("RKS Z-vector result consumption")
        if type(descriptor_diagnostics) is not DescriptorDiagnostics:
            raise RKSAdjointError(
                "the supplied descriptor diagnostics have an invalid type"
            )
        if type(adjoint) is not RKSAdjoint:
            raise RKSAdjointError(
                "the supplied RKS adjoint has an invalid type"
            )
        if type(adjoint.diagnostics) is not RKSAdjointDiagnostics:
            raise RKSAdjointError(
                "the supplied RKS adjoint diagnostics have an invalid type"
            )
        if adjoint is not self._trusted_adjoint:
            raise RKSAdjointError(
                "the supplied RKS adjoint was not produced by this DeePHF evaluation"
            )
        self._assert_trusted_rks_force_model_state(
            "RKS Z-vector result consumption"
        )
        if (
            type(self._trusted_adjoint_integrity) is not str
            or adjoint.integrity_fingerprint != self._trusted_adjoint_integrity
            or adjoint.integrity_fingerprint
            != rks_adjoint_integrity_fingerprint(adjoint)
        ):
            raise RKSAdjointError(
                "the supplied RKS adjoint failed its integrity check"
            )
        if not isinstance(sensitivity, np.ndarray):
            raise RKSAdjointError(
                "the supplied correction sensitivity has an invalid type"
            )
        expected_sensitivity_shape = (
            self.n_descriptor_atoms,
            self.n_descriptor_features,
        )
        if sensitivity.shape != expected_sensitivity_shape:
            raise RKSAdjointError(
                "the supplied correction sensitivity has an invalid shape"
            )
        if sensitivity.dtype != np.dtype(np.float64) or np.iscomplexobj(
            sensitivity
        ):
            raise RKSAdjointError(
                "the supplied correction sensitivity must use real numpy.float64"
            )
        if not np.isfinite(sensitivity).all() or sensitivity.flags.writeable:
            raise RKSAdjointError(
                "the supplied correction sensitivity must be finite and immutable"
            )
        if (
            type(self._trusted_adjoint_sensitivity_fingerprint) is not str
            or _array_fingerprint(sensitivity)
            != self._trusted_adjoint_sensitivity_fingerprint
        ):
            raise RKSAdjointError(
                "the supplied correction sensitivity failed its integrity check"
            )
        if descriptor_diagnostics != self._trusted_adjoint_descriptor_diagnostics:
            raise RKSAdjointError(
                "the supplied descriptor diagnostics do not match this evaluation"
            )
        adapter = self._trusted_rks_adjoint_adapter
        if type(adapter) is not RKSAdjointAdapter:
            raise RKSAdjointError(
                "the trusted RKS adjoint adapter is unavailable"
            )
        if (
            type(self._trusted_rks_adjoint_controls) is not tuple
            or self._rks_adjoint_controls(adapter)
            != self._trusted_rks_adjoint_controls
        ):
            raise RKSAdjointError(
                "the trusted RKS adjoint controls changed after the solve"
            )
        expected_objective_ao_potential = self._correction_ao_potential(
            sensitivity
        )
        adapter.audit_adjoint(adjoint, expected_objective_ao_potential)
        self._assert_science_state("RKS Z-vector result audit")
        self._assert_trusted_rks_force_model_state(
            "RKS Z-vector result audit"
        )
        return descriptor_diagnostics, sensitivity, adjoint

    def nuc_grad_method(self, *, backend="direct", **backend_options):
        """Build one explicitly selected strict RKS gradient backend."""
        if type(backend) is not str or backend not in {"direct", "zvector"}:
            raise ValueError(
                "RKS gradient backend must be 'direct' or 'zvector'"
            )
        if backend == "direct":
            _validated_backend_options(
                self.response_options,
                backend_options,
                _DIRECT_RESPONSE_OPTIONS,
                backend,
            )
            from .rks_gradient import RKSDeePHFGradients

            return RKSDeePHFGradients(
                self,
                response_options=backend_options,
            )
        _validated_backend_options(
            self.adjoint_options,
            backend_options,
            _RKS_ZVECTOR_OPTIONS,
            backend,
        )
        from .rks_zvector import RKSDeePHFZVectorGradients

        return RKSDeePHFZVectorGradients(
            self,
            adjoint_options=backend_options,
        )


__all__ = ["RKSDeePHF"]
