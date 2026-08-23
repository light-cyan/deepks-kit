"""Perturbative DeePHF energy method for a strict native UHF reference."""

import numpy as np

from deepks.descriptor import DescriptorDiagnostics

from .capabilities import (
    DeePHFCapabilityError,
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
from .pyscf_uhf import (
    UHFAdjoint,
    UHFAdjointAdapter,
    UHFAdjointDiagnostics,
    UHFAdjointError,
    UHFResponse,
    UHFResponseAdapter,
    UHFResponseDiagnostics,
    UHFResponseError,
    uhf_adjoint_integrity_fingerprint,
    uhf_reference_fingerprint,
    validate_uhf_reference,
)


_UHF_ZVECTOR_OPTIONS = frozenset(
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
        self._trusted_uhf_adjoint_adapter = None
        self._trusted_uhf_adjoint_controls = None

    def _clear_trusted_adjoint(self) -> None:
        """Clear every UHF correction-specific adjoint provenance binding."""
        super()._clear_trusted_adjoint()
        self._trusted_uhf_adjoint_adapter = None
        self._trusted_uhf_adjoint_controls = None

    @staticmethod
    def _uhf_adjoint_controls(adapter) -> tuple:
        return tuple(
            (name, getattr(adapter, name))
            for name in sorted(_UHF_ZVECTOR_OPTIONS)
        )

    def _assert_trusted_uhf_force_model_state(self, boundary: str) -> None:
        fingerprint = self._trusted_adjoint_model_fingerprint
        if type(fingerprint) is not str:
            raise UHFAdjointError(
                "the UHF Z-vector force-model provenance is unavailable"
            )
        self._assert_force_model_state(fingerprint, boundary)

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

    def adjoint(self, **adjoint_options) -> UHFAdjoint:
        """Solve one audited correction-specific coupled UHF adjoint."""
        self._clear_trusted_adjoint()
        try:
            inputs = self._zvector_inputs(adjoint_options)
            return self._validate_zvector_inputs(*inputs)[2]
        except Exception:
            self._clear_trusted_adjoint()
            raise

    def _zvector_inputs(self, adjoint_options):
        """Evaluate one internally consistent UHF sensitivity and adjoint."""
        self._clear_trusted_adjoint()
        try:
            self._assert_science_state("UHF Z-vector input evaluation")
            validate_force_model(self.model)
            model_fingerprint = force_model_fingerprint(self.model)
            self._validate_reference_object(self.reference)
            self._assert_science_state("UHF Z-vector reference validation")
            sensitivity = _immutable_array(self.correction_sensitivity())
            self._assert_force_model_state(
                model_fingerprint,
                "UHF Z-vector model sensitivity evaluation",
            )
            self._validate_science_state(
                "UHF Z-vector model sensitivity evaluation"
            )
            descriptor_diagnostics = (
                self._validate_force_compatibility_with_sensitivity(sensitivity)
            )
            self._assert_science_state("UHF Z-vector descriptor validation")
            self._assert_force_model_state(
                model_fingerprint,
                "UHF Z-vector descriptor validation",
            )
            options = _validated_backend_options(
                self.adjoint_options,
                adjoint_options,
                _UHF_ZVECTOR_OPTIONS,
                "zvector",
            )
            objective_ao_potential = self._correction_ao_potential(sensitivity)
            self._validate_science_state(
                "UHF Z-vector AO potential construction"
            )
            self._assert_force_model_state(
                model_fingerprint,
                "UHF Z-vector AO potential construction",
            )
            adapter = UHFAdjointAdapter(self.reference, **options)
            adjoint = adapter.solve(objective_ao_potential)
            self._assert_science_state("UHF Z-vector adjoint construction")
            self._assert_force_model_state(
                model_fingerprint,
                "UHF Z-vector adjoint construction",
            )
            self._trusted_adjoint = adjoint
            self._trusted_adjoint_integrity = adjoint.integrity_fingerprint
            self._trusted_adjoint_sensitivity_fingerprint = _array_fingerprint(
                sensitivity
            )
            self._trusted_adjoint_descriptor_diagnostics = descriptor_diagnostics
            self._trusted_adjoint_model_fingerprint = model_fingerprint
            self._trusted_uhf_adjoint_adapter = adapter
            self._trusted_uhf_adjoint_controls = self._uhf_adjoint_controls(
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
        """Audit one trusted UHF Z-vector tuple before consuming gradients."""
        self._assert_science_state("UHF Z-vector result consumption")
        if type(descriptor_diagnostics) is not DescriptorDiagnostics:
            raise UHFAdjointError(
                "the supplied descriptor diagnostics have an invalid type"
            )
        if type(adjoint) is not UHFAdjoint:
            raise UHFAdjointError(
                "the supplied UHF adjoint has an invalid type"
            )
        if type(adjoint.diagnostics) is not UHFAdjointDiagnostics:
            raise UHFAdjointError(
                "the supplied UHF adjoint diagnostics have an invalid type"
            )
        if adjoint is not self._trusted_adjoint:
            raise UHFAdjointError(
                "the supplied UHF adjoint was not produced by this evaluation"
            )
        self._assert_trusted_uhf_force_model_state(
            "UHF Z-vector result consumption"
        )
        if (
            type(self._trusted_adjoint_integrity) is not str
            or adjoint.integrity_fingerprint != self._trusted_adjoint_integrity
            or adjoint.integrity_fingerprint
            != uhf_adjoint_integrity_fingerprint(adjoint)
        ):
            raise UHFAdjointError(
                "the supplied UHF adjoint failed its integrity check"
            )
        if not isinstance(sensitivity, np.ndarray):
            raise UHFAdjointError(
                "the supplied correction sensitivity has an invalid type"
            )
        expected_shape = (
            self.n_descriptor_atoms,
            self.n_descriptor_features,
        )
        if sensitivity.shape != expected_shape:
            raise UHFAdjointError(
                "the supplied correction sensitivity has an invalid shape"
            )
        if sensitivity.dtype != np.dtype(np.float64) or np.iscomplexobj(
            sensitivity
        ):
            raise UHFAdjointError(
                "the supplied correction sensitivity must use real numpy.float64"
            )
        if not np.isfinite(sensitivity).all() or sensitivity.flags.writeable:
            raise UHFAdjointError(
                "the supplied correction sensitivity must be finite and immutable"
            )
        if (
            type(self._trusted_adjoint_sensitivity_fingerprint) is not str
            or _array_fingerprint(sensitivity)
            != self._trusted_adjoint_sensitivity_fingerprint
        ):
            raise UHFAdjointError(
                "the supplied correction sensitivity failed its integrity check"
            )
        if descriptor_diagnostics != self._trusted_adjoint_descriptor_diagnostics:
            raise UHFAdjointError(
                "the supplied descriptor diagnostics do not match this evaluation"
            )
        adapter = self._trusted_uhf_adjoint_adapter
        if type(adapter) is not UHFAdjointAdapter:
            raise UHFAdjointError(
                "the trusted UHF adjoint adapter is unavailable"
            )
        if (
            type(self._trusted_uhf_adjoint_controls) is not tuple
            or self._uhf_adjoint_controls(adapter)
            != self._trusted_uhf_adjoint_controls
        ):
            raise UHFAdjointError(
                "the trusted UHF adjoint controls changed after the solve"
            )
        expected_objective = self._correction_ao_potential(sensitivity)
        adapter.audit_adjoint(adjoint, expected_objective)
        self._assert_science_state("UHF Z-vector result audit")
        self._assert_trusted_uhf_force_model_state(
            "UHF Z-vector result audit"
        )
        return descriptor_diagnostics, sensitivity, adjoint

    def nuc_grad_method(self, *, backend="direct", **backend_options):
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
        )
