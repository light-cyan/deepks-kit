"""Perturbative DeePHF energy method composed around a native reference."""

from collections.abc import Mapping
import hashlib
from contextlib import contextmanager
import weakref

import numpy as np
import torch

from deepks.descriptor import (
    AtomicDensityDescriptor,
    spin_summed_ao_density,
    validate_differentiability,
)
from deepks.model.model import CorrNet

from .capabilities import (
    DeePHFCapabilityError,
    begin_reference_validation_transaction,
    end_reference_validation_transaction,
    science_state_transaction,
    validate_force_model,
    validate_model,
    validate_model_output,
)
from .gradient import _validate_atom_indices
from .pyscf_rhf import (
    RHFAdjoint,
    RHFAdjointAdapter,
    RHFResponse,
    RHFResponseAdapter,
    RHFResponseError,
    molecule_science_fingerprint,
    reference_fingerprint,
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
        "coordinate_block_size",
    }
)
_ZVECTOR_OPTIONS = frozenset(
    {
        "residual_tolerance",
        "orbital_gap_tolerance",
        "objective_symmetry_tolerance",
        "max_cycle",
        "krylov_restart",
    }
)
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
        self._science_transaction_depth = 0
        self._science_transaction_fingerprint = None
        self.response_options = dict(response_options or {})
        if adjoint_options is None:
            adjoint_options = {}
        elif not isinstance(adjoint_options, Mapping):
            raise TypeError("adjoint_options must be a mapping")
        self.adjoint_options = dict(adjoint_options)
        self._trusted_response_integrities = {}
        validate_model(
            self.model,
            self._descriptor.projector_basis,
            self._descriptor.n_features,
        )
        self._validated_model_output()
        self.e_base = None
        self.e_corr = None
        self.e_tot = None

    def _seal_response(self, response) -> None:
        self._trusted_response_integrities[id(response)] = (
            weakref.ref(response),
            response.integrity_fingerprint,
        )
        while len(self._trusted_response_integrities) > 32:
            self._trusted_response_integrities.pop(next(iter(self._trusted_response_integrities)))

    def _is_sealed_response(self, response) -> bool:
        sealed = self._trusted_response_integrities.get(id(response))
        return (
            sealed is not None
            and sealed[0]() is response
            and sealed[1] == response.integrity_fingerprint
        )

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
        self._latest_reference_state_fingerprint = (
            self._reference_state_fingerprint(self.reference)
        )
        digest.update(self._latest_reference_state_fingerprint.encode("ascii"))
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
        if self._science_transaction_depth:
            return self._science_transaction_fingerprint
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

    @contextmanager
    def _science_state_transaction(self):
        outermost = self._science_transaction_depth == 0
        reference_token = None
        if outermost:
            self._science_transaction_fingerprint = self._assert_science_state(
                "calculation entry"
            )
            reference_token = begin_reference_validation_transaction(
                self.reference,
                self._latest_reference_state_fingerprint,
            )
        self._science_transaction_depth += 1
        try:
            yield self._science_transaction_fingerprint
        finally:
            self._science_transaction_depth -= 1
            if outermost:
                end_reference_validation_transaction(reference_token)
                self._science_transaction_fingerprint = None
                self._assert_science_state("calculation exit")

    def _validate_science_state(self, boundary: str) -> str:
        self._assert_science_state(boundary)
        if self._science_transaction_depth:
            return self._science_transaction_fingerprint
        self._validate_reference_object(self.reference)
        return self._assert_science_state(boundary)

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

    def dq_dR_explicit(self, atom_indices=None):
        atom_indices = _validate_atom_indices(self.mol, atom_indices)
        with torch.enable_grad():
            return self._descriptor.dq_dR_explicit(
                self.ao_density(),
                raw_atom_indices=atom_indices,
            )

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

    @science_state_transaction
    def correction_sensitivity(self) -> np.ndarray:
        """Return one autograd derivative of the scalar correction with respect to q."""
        return self._correction_sensitivity(self._descriptor_values_tensor())

    def _correction_sensitivity(self, values: torch.Tensor) -> np.ndarray:
        self._assert_science_state("model sensitivity evaluation")
        validate_force_model(self.model)
        validate_model(
            self.model,
            self._descriptor.projector_basis,
            self._descriptor.n_features,
        )
        if self.model is None:
            sensitivity = torch.zeros_like(values)
        else:
            evaluation_values = values.detach().requires_grad_(True)
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
        self._assert_science_state("model sensitivity evaluation")
        return sensitivity.detach().cpu().numpy()

    def _correction_ao_potential(
        self,
        sensitivity: np.ndarray,
        dq_dP: np.ndarray | None = None,
    ) -> np.ndarray:
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
            self.dq_dP() if dq_dP is None else dq_dP,
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

    @science_state_transaction
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

    @science_state_transaction
    def validate_force_compatibility(self, **tolerances):
        """Validate ordered-eigenvalue and model-sensitivity force semantics."""
        self._validate_reference_object(self.reference)
        return self._force_inputs(**tolerances)[0]

    def _force_inputs(self, **tolerances):
        values = self._descriptor_values_tensor()
        sensitivity = self._correction_sensitivity(values)
        diagnostics = self._validate_force_compatibility_with_sensitivity(
            sensitivity,
            descriptor_values=values,
            **tolerances,
        )
        return diagnostics, sensitivity

    def _validate_force_compatibility_with_sensitivity(
        self,
        sensitivity,
        descriptor_values=None,
        **tolerances,
    ):
        values = (
            self._descriptor_values_tensor()
            if descriptor_values is None
            else descriptor_values
        )
        n_occupied = self._descriptor_rank_bound()
        diagnostics = validate_differentiability(
            values.detach().cpu().numpy(),
            self._descriptor.shell_sizes,
            n_occupied,
            sensitivity,
            **tolerances,
        )
        return diagnostics

    @science_state_transaction
    def response(self, **response_options) -> RHFResponse:
        """Solve the audited complete first-order RHF density response."""
        self.validate_force_compatibility()
        options = _validated_backend_options(
            self.response_options,
            response_options,
            _DIRECT_RESPONSE_OPTIONS,
            "direct",
        )
        options.pop("coordinate_block_size", None)
        response = RHFResponseAdapter(self.reference, **options).solve()
        self._seal_response(response)
        return response

    @science_state_transaction
    def adjoint(self, **adjoint_options) -> RHFAdjoint:
        """Solve one audited correction-specific RHF scalar adjoint."""
        return self._zvector_inputs(adjoint_options)[2]

    def _zvector_inputs(self, adjoint_options):
        """Build one sensitivity and one scalar adjoint."""
        self._assert_science_state("Z-vector input evaluation")
        self._validate_reference_object(self.reference)
        descriptor_diagnostics, sensitivity = self._force_inputs()
        sensitivity = _immutable_array(sensitivity)
        options = _validated_backend_options(
            self.adjoint_options,
            adjoint_options,
            _ZVECTOR_OPTIONS,
            "zvector",
        )
        objective_ao_potential = self._correction_ao_potential(sensitivity)
        adjoint = RHFAdjointAdapter(self.reference, **options).solve(
            objective_ao_potential
        )
        return descriptor_diagnostics, sensitivity, adjoint

    def _validate_response(self, response: RHFResponse) -> RHFResponse:
        """Return the latest response sealed by this method."""
        self._assert_science_state("RHF response consumption")
        self._validate_reference_object(self.reference)
        if type(response) is not RHFResponse:
            raise RHFResponseError("the supplied RHF response has an invalid type")
        if not self._is_sealed_response(response):
            raise RHFResponseError(
                "the supplied RHF response was not produced by this method"
            )
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

    @science_state_transaction
    def dq_dR_response(
        self,
        response: RHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return the descriptor derivative generated by the RHF density response."""
        density_response = self.first_order_density(
            response=response,
            **response_options,
        )
        return np.einsum(
            "apij,bxij->bxap",
            self.dq_dP(),
            density_response,
        )

    @science_state_transaction
    def dq_dR_relaxed(
        self,
        response: RHFResponse | None = None,
        **response_options,
    ) -> np.ndarray:
        """Return dq/dR including explicit projector motion and RHF response."""
        if response is None:
            response = self.response(**response_options)
            response_options = {}
        else:
            self.validate_force_compatibility()
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
