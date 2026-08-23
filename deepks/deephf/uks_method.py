"""Perturbative DeePHF energy method for a strict finite-grid UKS reference."""

import numpy as np

from .capabilities import science_state_transaction
from .method import (
    _DIRECT_RESPONSE_OPTIONS,
    _immutable_array,
    _validated_backend_options,
)
from .pyscf_uks import (
    UKSAdjoint,
    UKSAdjointAdapter,
    UKSResponse,
    UKSResponseAdapter,
    UKSResponseDiagnostics,
    UKSResponseError,
    uks_reference_fingerprint,
    validate_uks_reference,
)
from .uhf_method import UHFDeePHF


_UKS_ZVECTOR_OPTIONS = frozenset(
    {
        "residual_tolerance",
        "invariant_tolerance",
        "orbital_gap_tolerance",
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

    @science_state_transaction
    def response(self, **response_options) -> UKSResponse:
        """Solve the audited complete finite-grid UKS density response."""
        self.validate_force_compatibility()
        return self._solve_response(response_options)

    def _solve_response(self, response_options, atom_indices=None) -> UKSResponse:
        """Solve one response after the caller has validated descriptor semantics."""
        options = _validated_backend_options(
            self.response_options,
            response_options,
            _DIRECT_RESPONSE_OPTIONS,
            "direct",
        )
        adapter = UKSResponseAdapter(self.reference, **options)
        response = adapter.solve(atom_indices=atom_indices)
        self._seal_response(response)
        return response

    def _validate_response(self, response: UKSResponse) -> UKSResponse:
        """Audit one response produced by this exact UKS method."""
        self._assert_science_state("UKS response consumption")
        self._validate_reference_object(self.reference)
        if (
            type(response) is not UKSResponse
            or type(response.diagnostics) is not UKSResponseDiagnostics
        ):
            raise UKSResponseError("the supplied UKS response has an invalid type")
        if not self._is_sealed_response(response):
            raise UKSResponseError(
                "the supplied UKS response was not produced by this UKS DeePHF method"
            )
        return response

    @science_state_transaction
    def adjoint(self, **adjoint_options) -> UKSAdjoint:
        """Solve one audited correction-specific coupled UKS adjoint."""
        return self._zvector_inputs(adjoint_options)[2]

    def _zvector_inputs(self, adjoint_options, atom_indices=None):
        """Build one UKS sensitivity and one scalar adjoint."""
        self._validate_reference_object(self.reference)
        descriptor_diagnostics, sensitivity = self._force_inputs()
        sensitivity = _immutable_array(sensitivity)
        options = _validated_backend_options(
            self.adjoint_options,
            adjoint_options,
            _UKS_ZVECTOR_OPTIONS,
            "zvector",
        )
        objective = self._correction_ao_potential(sensitivity)
        adjoint = UKSAdjointAdapter(self.reference, **options).solve(
            objective,
            atom_indices=atom_indices,
        )
        return descriptor_diagnostics, sensitivity, adjoint

    def nuc_grad_method(self, *, backend="direct", retain_details=True, **backend_options):
        """Build one explicitly selected finite-grid UKS gradient backend."""
        if type(backend) is not str or backend not in {"direct", "zvector"}:
            raise ValueError("UKS gradient backend must be 'direct' or 'zvector'")
        if backend == "direct":
            _validated_backend_options(self.response_options, backend_options, _DIRECT_RESPONSE_OPTIONS, backend)
            from .uks_gradient import UKSDeePHFGradients

            return UKSDeePHFGradients(
                self,
                response_options=backend_options,
                retain_details=retain_details,
            )
        _validated_backend_options(self.adjoint_options, backend_options, _UKS_ZVECTOR_OPTIONS, backend)
        from .uks_zvector import UKSDeePHFZVectorGradients

        return UKSDeePHFZVectorGradients(
            self,
            adjoint_options=backend_options,
            retain_details=retain_details,
        )


__all__ = ["UKSDeePHF"]
