"""Calculation-scoped DeePHF evaluation state and operation accounting."""

from collections import Counter

import numpy as np
import torch

from deepks.descriptor import spin_summed_ao_density, validate_differentiability

from .capabilities import (
    DeePHFCapabilityError,
    validate_force_model,
    validate_model_output,
)


class EvaluationContext:
    """Lazily evaluate and share all model-neutral and model-dependent inputs."""

    def __init__(self, method, state_token: str, counters=None):
        self.method = method
        self.state_token = state_token
        self.counters = Counter() if counters is None else counters
        self._density = None
        self._spin_density = None
        self._workspace = None
        self._model_values = None
        self._model_output = None
        self._sensitivity = None
        self._diagnostics = {}

    def count(self, operation: str) -> None:
        self.counters[operation] += 1

    @property
    def density(self) -> np.ndarray:
        if self._density is None:
            raw_density = np.asarray(self.method.reference.make_rdm1())
            if raw_density.ndim == 3:
                self._spin_density = raw_density
            self._density = spin_summed_ao_density(raw_density)
            self.count("ao_density_constructions")
        return self._density

    @property
    def spin_density(self) -> np.ndarray:
        if self._spin_density is None:
            _density = self.density
        if self._spin_density is None:
            raise DeePHFCapabilityError(
                "the bound reference does not provide spin-resolved AO density"
            )
        return self._spin_density

    @property
    def workspace(self):
        if self._workspace is None:
            self._workspace = self.method._descriptor.derivative_workspace(
                self.density,
                operation_hook=self.count,
            )
        return self._workspace

    @property
    def descriptor_values(self) -> torch.Tensor:
        if self._model_values is None:
            self.count("descriptor_evaluations")
            values = self.workspace.descriptor_values.detach()
            self._model_values = values.requires_grad_(self.method.model is not None)
        return self._model_values

    @property
    def model_output(self) -> torch.Tensor:
        if self._model_output is None:
            if self.method.model is None:
                self._model_output = torch.zeros((), dtype=torch.float64)
            else:
                self.count("model_forwards")
                with torch.enable_grad():
                    self._model_output = validate_model_output(
                        self.method.model,
                        self.descriptor_values,
                    )
        return self._model_output

    @property
    def correction_energy(self) -> float:
        energy = (
            float(self.model_output.detach().cpu().item())
            + self.method._validated_element_constant()
        )
        if not np.isfinite(energy):
            raise DeePHFCapabilityError(
                "the complete correction energy must be finite"
            )
        return energy

    @property
    def sensitivity(self) -> np.ndarray:
        if self._sensitivity is None:
            validate_force_model(self.method.model)
            values = self.descriptor_values
            if self.method.model is None:
                sensitivity = torch.zeros_like(values)
            else:
                energy = self.model_output
                if energy.requires_grad:
                    (sensitivity,) = torch.autograd.grad(
                        energy,
                        values,
                        torch.ones_like(energy),
                        allow_unused=True,
                    )
                else:
                    sensitivity = None
                if sensitivity is None:
                    sensitivity = torch.zeros_like(values)
            if sensitivity.shape != values.shape:
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
            self._sensitivity = sensitivity.detach().cpu().numpy()
        return self._sensitivity

    def force_inputs(self, **tolerances):
        key = tuple(sorted(tolerances.items()))
        diagnostics = self._diagnostics.get(key)
        if diagnostics is None:
            method = self.method
            diagnostics = validate_differentiability(
                self.descriptor_values.detach().cpu().numpy(),
                method._descriptor.shell_sizes,
                method._descriptor_rank_bound(),
                self.sensitivity,
                **tolerances,
            )
            self._diagnostics[key] = diagnostics
        return diagnostics, self.sensitivity

    @property
    def sensitivity_is_zero(self) -> bool:
        return not np.any(self.sensitivity)


__all__ = ["EvaluationContext"]
