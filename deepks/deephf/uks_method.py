"""Perturbative DeePHF energy method for a strict finite-grid UKS reference."""


from .method import (
    _DIRECT_RESPONSE_OPTIONS,
    _validated_backend_options,
)
from .pyscf_uks_reference import (
    UKSResponse,
    UKSResponseDiagnostics,
    UKSResponseError,
    uks_reference_fingerprint,
    validate_uks_reference,
)
from .pyscf_uks_response import UKSAdjointAdapter, UKSResponseAdapter
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

    _adjoint_adapter_type = UKSAdjointAdapter
    _response_adapter_type = UKSResponseAdapter
    _zvector_options = _UKS_ZVECTOR_OPTIONS

    @staticmethod
    def _validate_reference_object(reference):
        return validate_uks_reference(reference)

    @staticmethod
    def _reference_state_fingerprint(reference, *, use_transaction=True) -> str:
        return uks_reference_fingerprint(
            reference,
            use_transaction=use_transaction,
        )

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
