"""Perturbative DeePHF energy method for a strict native UHF reference."""

import numpy as np

from .capabilities import (
    DeePHFCapabilityError,
    science_state_transaction,
)
from .method import (
    DeePHF,
    _DIRECT_RESPONSE_OPTIONS,
    _immutable_array,
    _validated_backend_options,
)
from .pyscf_uhf import (
    UHFAdjoint,
    UHFAdjointAdapter,
    UHFResponse,
    UHFResponseAdapter,
    UHFResponseDiagnostics,
    UHFResponseError,
    uhf_reference_fingerprint,
    validate_uhf_reference,
)


_UHF_ZVECTOR_OPTIONS = frozenset(
    {
        "residual_tolerance",
        "invariant_tolerance",
        "orbital_gap_tolerance",
        "objective_symmetry_tolerance",
        "max_cycle",
        "krylov_restart",
    }
)


class UHFDeePHF(DeePHF):
    """Evaluate a perturbative correction around one strict UHF reference."""

    @staticmethod
    def _validate_reference_object(reference):
        return validate_uhf_reference(reference)

    @staticmethod
    def _reference_state_fingerprint(reference) -> str:
        return uhf_reference_fingerprint(reference)

    def _descriptor_rank_bound(self) -> int:
        occupied_count = int(np.count_nonzero(self.reference.mo_occ > 0))
        return min(int(self.mol.nao), occupied_count)

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

    def spin_ao_density(self) -> np.ndarray:
        """Return the native alpha and beta AO densities."""
        self._assert_science_state("spin-resolved AO density evaluation")
        density = np.asarray(self.reference.make_rdm1())
        expected_shape = (2, self.mol.nao, self.mol.nao)
        if density.shape != expected_shape:
            raise DeePHFCapabilityError(
                "the UHF spin-resolved AO density shape is invalid"
            )
        if density.dtype != np.dtype(np.float64) or np.iscomplexobj(density):
            raise DeePHFCapabilityError(
                "the UHF spin-resolved AO density must use real numpy.float64"
            )
        if not np.isfinite(density).all():
            raise DeePHFCapabilityError(
                "the UHF spin-resolved AO density must be finite"
            )
        self._assert_science_state("spin-resolved AO density evaluation")
        return density

    def dq_dR_explicit_spin(self, atom_indices=None) -> np.ndarray:
        """Return additive alpha and beta components of fixed-density dq/dR."""
        from .gradient import _validate_atom_indices

        atom_indices = _validate_atom_indices(self.mol, atom_indices)
        spin_density = self.spin_ao_density()
        total_density = spin_density.sum(axis=0)
        components = np.stack(
            [
                self._descriptor.dq_dR_explicit_component(
                    total_density,
                    spin_density[spin_index],
                    raw_atom_indices=atom_indices,
                )
                for spin_index in range(2)
            ]
        )
        return components

    @science_state_transaction
    def response(self, **response_options) -> UHFResponse:
        """Solve the audited complete coupled UHF density response."""
        self.validate_force_compatibility()
        return self._solve_response(response_options)

    def _solve_response(
        self,
        response_options,
        atom_indices=None,
        result_mode="response",
    ):
        """Solve one response after the caller has validated descriptor semantics."""
        options = _validated_backend_options(
            self.response_options,
            response_options,
            _DIRECT_RESPONSE_OPTIONS,
            "direct",
        )
        adapter = UHFResponseAdapter(self.reference, **options)
        if result_mode == "gradient":
            return adapter._solve_for_gradient(atom_indices=atom_indices)
        if result_mode == "partitions":
            response, density_partitions = adapter._solve_with_density_partitions(
                atom_indices=atom_indices
            )
        else:
            response = adapter.solve(atom_indices=atom_indices)
        self._seal_response(response)
        return (
            (response, density_partitions)
            if result_mode == "partitions"
            else response
        )

    def _validate_response(self, response: UHFResponse) -> UHFResponse:
        """Return a response produced by this exact UHF method."""
        self._assert_science_state("UHF response consumption")
        self._validate_reference_object(self.reference)
        if type(response) is not UHFResponse:
            raise UHFResponseError("the supplied UHF response has an invalid type")
        if type(response.diagnostics) is not UHFResponseDiagnostics:
            raise UHFResponseError(
                "the supplied UHF response diagnostics have an invalid type"
            )
        if not self._is_sealed_response(response):
            raise UHFResponseError(
                "the supplied UHF response was not produced by this method"
            )
        return response

    def first_order_spin_density(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return alpha and beta first-order AO density responses."""
        if response is not None and response_options:
            raise ValueError("response and response_options are mutually exclusive")
        if response is None:
            response = self.response(**response_options)
        response = self._validate_response(response)
        return np.stack(
            (response.alpha_density_response, response.beta_density_response)
        )

    def first_order_density(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return the spin-summed first-order AO density response."""
        if response is not None and response_options:
            raise ValueError("response and response_options are mutually exclusive")
        if response is None:
            response = self.response(**response_options)
        return self._validate_response(response).total_density_response

    @science_state_transaction
    def dq_dR_response_spin(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return alpha and beta descriptor-response contributions."""
        spin_density_response = self.first_order_spin_density(
            response=response,
            **response_options,
        )
        result = np.einsum(
            "apij,sbxij->sbxap",
            self.dq_dP(),
            spin_density_response,
        )
        if not np.isfinite(result).all():
            raise UHFResponseError(
                "the UHF spin-resolved descriptor response is nonfinite"
            )
        return result

    def dq_dR_response(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return the spin-summed descriptor response contribution."""
        return self.dq_dR_response_spin(
            response=response,
            **response_options,
        ).sum(axis=0)

    @science_state_transaction
    def dq_dR_relaxed_spin(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return additive alpha and beta relaxed descriptor derivatives."""
        if response is None:
            response = self.response(**response_options)
            response_options = {}
        else:
            self.validate_force_compatibility()
        return self.dq_dR_explicit_spin() + self.dq_dR_response_spin(
            response=response,
            **response_options,
        )

    @science_state_transaction
    def dq_dR_relaxed(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return complete relaxed dq/dR for the spin-summed descriptor."""
        return self.dq_dR_relaxed_spin(
            response=response,
            **response_options,
        ).sum(axis=0)

    @science_state_transaction
    def adjoint(self, **adjoint_options) -> UHFAdjoint:
        """Solve one audited correction-specific coupled UHF adjoint."""
        return self._zvector_inputs(adjoint_options)[2]

    def _zvector_inputs(self, adjoint_options, atom_indices=None):
        """Build one UHF sensitivity and one scalar adjoint."""
        self._validate_reference_object(self.reference)
        descriptor_diagnostics, sensitivity = self._force_inputs()
        sensitivity = _immutable_array(sensitivity)
        options = _validated_backend_options(
            self.adjoint_options,
            adjoint_options,
            _UHF_ZVECTOR_OPTIONS,
            "zvector",
        )
        objective = self._correction_ao_potential(sensitivity)
        adjoint = UHFAdjointAdapter(self.reference, **options).solve(
            objective,
            atom_indices=atom_indices,
        )
        return descriptor_diagnostics, sensitivity, adjoint

    def nuc_grad_method(self, *, backend="direct", retain_details=True, **backend_options):
        """Build one explicitly selected strict UHF gradient backend."""
        if type(backend) is not str or backend not in {"direct", "zvector"}:
            raise ValueError(
                "UHF gradient backend must be 'direct' or 'zvector'"
            )
        if backend == "direct":
            _validated_backend_options(
                self.response_options,
                backend_options,
                _DIRECT_RESPONSE_OPTIONS,
                backend,
            )
            from .uhf_gradient import UHFDeePHFGradients

            return UHFDeePHFGradients(
                self,
                response_options=backend_options,
                retain_details=retain_details,
            )
        _validated_backend_options(
            self.adjoint_options,
            backend_options,
            _UHF_ZVECTOR_OPTIONS,
            backend,
        )
        from .uhf_zvector import UHFDeePHFZVectorGradients

        return UHFDeePHFZVectorGradients(
            self,
            adjoint_options=backend_options,
            retain_details=retain_details,
        )
