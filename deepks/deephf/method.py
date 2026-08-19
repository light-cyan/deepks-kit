"""Perturbative DeePHF energy method composed around a native reference."""

import numpy as np
import torch

from deepks.descriptor import AtomicDensityDescriptor, validate_differentiability
from deepks.model.evaluate import descriptor_sensitivity
from deepks.model.model import CorrNet

from .capabilities import (
    DeePHFCapabilityError,
    validate_model,
    validate_model_output,
    validate_reference,
)
from .pyscf_rhf import (
    RHFResponse,
    RHFResponseAdapter,
    RHFResponseDiagnostics,
    RHFResponseError,
    reference_fingerprint,
    response_integrity_fingerprint,
)


class DeePHF:
    """Evaluate a perturbative correction without modifying the RHF reference."""

    def __init__(
        self,
        reference,
        model,
        projector_basis=None,
        device="cpu",
        response_options=None,
    ):
        self.reference = validate_reference(reference)
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
        self.response_options = dict(response_options or {})
        self._trusted_response = None
        self._trusted_response_integrity = None
        validate_model(
            self.model,
            self._descriptor.projector_basis,
            self._descriptor.n_features,
        )
        self._validated_model_output()
        self.e_base = None
        self.e_corr = None
        self.e_tot = None

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
        return np.asarray(self.reference.make_rdm1())

    def projected_density(self, flatten=False):
        return self._descriptor.projected_density(
            self.ao_density(),
            flatten=flatten,
        )

    def descriptor(self):
        return self._descriptor.descriptor(self.ao_density())

    def dq_dP(self):
        return self._descriptor.dq_dP(self.ao_density())

    def dq_dR_explicit(self):
        return self._descriptor.dq_dR_explicit(self.ao_density())

    def _descriptor_values_tensor(self) -> torch.Tensor:
        return self._descriptor.torch_descriptor(self.ao_density())

    def _validated_model_output(self) -> torch.Tensor:
        validate_model(
            self.model,
            self._descriptor.projector_basis,
            self._descriptor.n_features,
        )
        return validate_model_output(
            self.model,
            self._descriptor_values_tensor(),
        )

    def correction_sensitivity(self) -> np.ndarray:
        """Return the validated derivative of the scalar correction with respect to q."""
        values = self._descriptor_values_tensor()
        self._validated_model_output()
        if self.model is None:
            sensitivity = torch.zeros_like(values)
        else:
            sensitivity = descriptor_sensitivity(self.model, values)
        if not torch.isfinite(sensitivity).all():
            raise DeePHFCapabilityError(
                "the correction model descriptor sensitivity must be finite"
            )
        return sensitivity.detach().cpu().numpy()

    def correction_energy(self):
        if self.model is None:
            return 0.0
        tensor_energy = self._validated_model_output()
        energy = float(tensor_energy.detach().cpu().item())
        get_element_constant = getattr(self.model, "get_elem_const", None)
        if get_element_constant is not None:
            element_constant = get_element_constant(
                [int(charge) for charge in self.mol.atom_charges()]
            )
            element_constant = np.asarray(element_constant)
            if element_constant.size != 1:
                raise DeePHFCapabilityError(
                    "the correction model element constant must be scalar"
                )
            if np.iscomplexobj(element_constant) or not np.isfinite(
                element_constant
            ).all():
                raise DeePHFCapabilityError(
                    "the correction model element constant must be real and finite"
                )
            energy += float(element_constant.reshape(-1)[0])
        if not np.isfinite(energy):
            raise DeePHFCapabilityError(
                "the complete correction energy must be finite"
            )
        return energy

    def validate_force_compatibility(self, **tolerances):
        """Validate ordered-eigenvalue and model-sensitivity force semantics."""
        validate_reference(self.reference)
        values = self._descriptor_values_tensor()
        sensitivity = self.correction_sensitivity()
        n_occupied = int(np.count_nonzero(self.reference.mo_occ > 0))
        return validate_differentiability(
            values.detach().cpu().numpy(),
            self._descriptor.shell_sizes,
            n_occupied,
            sensitivity,
            **tolerances,
        )

    def response(self, **response_options) -> RHFResponse:
        """Solve the audited complete first-order RHF density response."""
        self.validate_force_compatibility()
        options = {**self.response_options, **response_options}
        response = RHFResponseAdapter(self.reference, **options).solve()
        self._trusted_response = response
        self._trusted_response_integrity = response.integrity_fingerprint
        return response

    def _validate_response(self, response: RHFResponse) -> RHFResponse:
        """Reject response data that are stale, foreign, mutable, or incomplete."""
        validate_reference(self.reference)
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

    def nuc_grad_method(self, **response_options):
        """Build the strict RHF DeePHF analytic nuclear-gradient driver."""
        from .gradient import RHFDeePHFGradients

        return RHFDeePHFGradients(self, response_options=response_options)

    def gradient(self, **response_options) -> np.ndarray:
        """Evaluate the strict analytic nuclear energy gradient."""
        return self.nuc_grad_method(**response_options).kernel()

    def forces(self, **response_options) -> np.ndarray:
        """Evaluate nuclear forces as the negative analytic gradient."""
        return -self.gradient(**response_options)

    def kernel(self):
        """Evaluate E_base + E_corr while leaving the reference unchanged."""
        validate_reference(self.reference)
        self.e_base = float(self.reference.e_tot)
        self.e_corr = self.correction_energy()
        self.e_tot = self.e_base + self.e_corr
        if not np.isfinite(self.e_tot):
            raise DeePHFCapabilityError("the total DeePHF energy must be finite")
        return self.e_tot
