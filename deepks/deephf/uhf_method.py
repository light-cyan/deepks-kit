"""Perturbative DeePHF energy method for a strict native UHF reference."""

from collections.abc import Mapping

import numpy as np

from .capabilities import DeePHFCapabilityError
from .method import DeePHF, _DIRECT_RESPONSE_OPTIONS, _validated_backend_options
from .pyscf_uhf import (
    UHFResponse,
    UHFResponseAdapter,
    UHFResponseDiagnostics,
    UHFResponseError,
    uhf_reference_fingerprint,
    validate_uhf_reference,
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
        if adjoint_options is not None:
            if not isinstance(adjoint_options, Mapping):
                raise TypeError("adjoint_options must be a mapping")
            if adjoint_options:
                raise DeePHFCapabilityError(
                    "UHF DeePHF does not provide an adjoint backend"
                )
        super().__init__(
            reference,
            model,
            projector_basis=projector_basis,
            device=device,
            response_options=response_options,
            adjoint_options={},
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

    def dq_dR_explicit_spin(self) -> np.ndarray:
        """Return additive alpha and beta components of fixed-density dq/dR."""
        spin_density = self.spin_ao_density()
        total_density = spin_density.sum(axis=0)
        components = np.stack(
            [
                self._descriptor.dq_dR_explicit_component(
                    total_density,
                    spin_density[spin_index],
                )
                for spin_index in range(2)
            ]
        )
        total = self.dq_dR_explicit()
        if not np.allclose(
            components.sum(axis=0),
            total,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise UHFResponseError(
                "the UHF explicit descriptor spin components are inconsistent"
            )
        return components

    def response(self, **response_options) -> UHFResponse:
        """Solve the audited complete coupled UHF density response."""
        self.validate_force_compatibility()
        options = _validated_backend_options(
            self.response_options,
            response_options,
            _DIRECT_RESPONSE_OPTIONS,
            "direct",
        )
        response = UHFResponseAdapter(self.reference, **options).solve()
        self._trusted_response = response
        self._trusted_response_integrity = response.integrity_fingerprint
        return response

    @staticmethod
    def _response_adapter_options(diagnostics: UHFResponseDiagnostics) -> dict:
        return {
            "cphf_tolerance": diagnostics.cphf_tolerance,
            "residual_tolerance": diagnostics.residual_tolerance,
            "invariant_tolerance": diagnostics.invariant_tolerance,
            "orbital_gap_tolerance": diagnostics.orbital_gap_tolerance,
            "max_cycle": diagnostics.max_cycle,
            "max_refinement_cycles": diagnostics.max_refinement_cycles,
            "level_shift": diagnostics.level_shift,
            "operator_stability_tolerance": (
                diagnostics.operator_stability_tolerance
            ),
            "operator_condition_tolerance": (
                diagnostics.operator_condition_tolerance
            ),
            "operator_symmetry_tolerance": (
                diagnostics.operator_symmetry_tolerance
            ),
            "operator_dimension_limit": diagnostics.operator_dimension_limit,
        }

    def _validate_response(self, response: UHFResponse) -> UHFResponse:
        """Rebuild and audit one supplied spin-resolved response."""
        self._validate_reference_object(self.reference)
        if type(response) is not UHFResponse:
            raise UHFResponseError("the supplied UHF response has an invalid type")
        if type(response.diagnostics) is not UHFResponseDiagnostics:
            raise UHFResponseError(
                "the supplied UHF response diagnostics have an invalid type"
            )
        adapter = UHFResponseAdapter(
            self.reference,
            **self._response_adapter_options(response.diagnostics),
        )
        adapter.audit_response_equations(response)
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

    def dq_dR_response_spin(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return alpha and beta descriptor-response contributions."""
        self.validate_force_compatibility()
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

    def dq_dR_relaxed_spin(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return additive alpha and beta relaxed descriptor derivatives."""
        if response is None:
            response = self.response(**response_options)
            response_options = {}
        return self.dq_dR_explicit_spin() + self.dq_dR_response_spin(
            response=response,
            **response_options,
        )

    def dq_dR_relaxed(
        self,
        response: UHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return complete relaxed dq/dR for the spin-summed descriptor."""
        relaxed_spin = self.dq_dR_relaxed_spin(
            response=response,
            **response_options,
        )
        relaxed = relaxed_spin.sum(axis=0)
        expected = self.dq_dR_explicit() + (
            relaxed_spin - self.dq_dR_explicit_spin()
        ).sum(axis=0)
        if not np.allclose(relaxed, expected, rtol=0.0, atol=1.0e-12):
            raise UHFResponseError(
                "the UHF relaxed descriptor partitions are inconsistent"
            )
        return relaxed

    def adjoint(self, **adjoint_options):
        """Reject unavailable unrestricted adjoint inference."""
        raise DeePHFCapabilityError(
            "UHF DeePHF does not provide an adjoint backend"
        )

    def _zvector_inputs(self, adjoint_options):
        raise DeePHFCapabilityError(
            "UHF DeePHF does not provide a Z-vector backend"
        )

    def nuc_grad_method(self, *, backend="direct", **backend_options):
        """Build the strict UHF direct analytic-gradient backend."""
        if type(backend) is not str or backend != "direct":
            raise DeePHFCapabilityError(
                "the UHF DeePHF gradient backend must be 'direct'"
            )
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
        )

