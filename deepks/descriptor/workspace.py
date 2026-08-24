"""Shared derivative workspace for one atomic-density descriptor evaluation."""

from collections.abc import Callable

import numpy as np
import torch

from deepks.array_utils import immutable_array


class DescriptorDerivativeWorkspace:
    """Own descriptor primitives that are reusable within one calculation."""

    def __init__(
        self,
        descriptor,
        ao_density,
        *,
        operation_hook: Callable[[str], None] | None = None,
    ):
        self.descriptor = descriptor
        self.density = descriptor.as_ao_density_tensor(ao_density)
        self._operation_hook = operation_hook
        self._projected_blocks = None
        self._eigenvalues = None
        self._eigenvalue_jacobians = None
        self._dq_dP = None

    def _count(self, operation: str) -> None:
        if self._operation_hook is not None:
            self._operation_hook(operation)

    @property
    def projected_blocks(self) -> tuple[torch.Tensor, ...]:
        if self._projected_blocks is None:
            self._count("projected_density_constructions")
            self._projected_blocks = tuple(
                torch.einsum(
                    "rap,rs,saq->apq",
                    overlap,
                    self.density,
                    overlap,
                )
                for overlap in self.descriptor.overlap_shells
            )
        else:
            self._count("cache_hits")
        return self._projected_blocks

    def _diagonalize(self) -> None:
        if self._eigenvalues is not None:
            self._count("cache_hits")
            return
        eigenpairs = tuple(torch.linalg.eigh(block) for block in self.projected_blocks)
        self._eigenvalues = tuple(values for values, _vectors in eigenpairs)
        self._eigenvectors = tuple(vectors for _values, vectors in eigenpairs)

    @property
    def eigenvalues(self) -> tuple[torch.Tensor, ...]:
        self._diagonalize()
        return self._eigenvalues

    @property
    def descriptor_values(self) -> torch.Tensor:
        return torch.cat(self.eigenvalues, dim=-1)

    @property
    def eigenvalue_jacobians(self) -> tuple[torch.Tensor, ...]:
        if self._eigenvalue_jacobians is None:
            self._diagonalize()
            self._count("shell_eigenvalue_jacobian_constructions")
            self._eigenvalue_jacobians = tuple(
                torch.einsum("apv,aqv->avpq", vectors, vectors)
                for vectors in self._eigenvectors
            )
        else:
            self._count("cache_hits")
        return self._eigenvalue_jacobians

    @property
    def derivative_overlap_shells(self) -> tuple[torch.Tensor, ...]:
        if self.descriptor._derivative_overlap_shells is None:
            self._count("derivative_overlap_integral_evaluations")
        else:
            self._count("cache_hits")
        return self.descriptor.derivative_overlap_shells()

    def projected_density(self, *, flatten=False):
        if flatten:
            values = torch.cat(
                [block.flatten(-2) for block in self.projected_blocks], dim=-1
            ).detach().cpu().numpy()
            return immutable_array(values, dtype=np.float64)
        return [
            immutable_array(block.detach().cpu().numpy(), dtype=np.float64)
            for block in self.projected_blocks
        ]

    @property
    def cached_dq_dP(self) -> np.ndarray:
        if self._dq_dP is None:
            ao_jacobians = [
                torch.einsum("rap,avpq,saq->avrs", overlap, jacobian, overlap)
                for overlap, jacobian in zip(
                    self.descriptor.overlap_shells,
                    self.eigenvalue_jacobians,
                    strict=True,
                )
            ]
            self._dq_dP = torch.cat(ao_jacobians, dim=1).detach().cpu().numpy()
        else:
            self._count("cache_hits")
        return self._dq_dP

    def dq_dP(self) -> np.ndarray:
        return immutable_array(self.cached_dq_dP, dtype=np.float64)

    def _coordinate_jacobians(self, motion_density, raw_atom_indices):
        from .derivatives import dD_dR_explicit

        return dD_dR_explicit(
            self.descriptor.mol,
            self.descriptor.as_ao_density_tensor(motion_density),
            self.descriptor.overlap_shells,
            self.derivative_overlap_shells,
            self.descriptor.descriptor_atom_indices,
            raw_atom_indices,
        )

    def dq_dR_explicit(self, *, motion_density=None, raw_atom_indices=None):
        motion_density = self.density if motion_density is None else motion_density
        coordinates = self._coordinate_jacobians(motion_density, raw_atom_indices)
        result = torch.cat(
            [
                torch.einsum("bxapq,avpq->bxav", coordinate, jacobian)
                for coordinate, jacobian in zip(
                    coordinates,
                    self.eigenvalue_jacobians,
                    strict=True,
                )
            ],
            dim=-1,
        )
        return immutable_array(result.detach().cpu().numpy(), dtype=np.float64)

    def correction_derivatives(self, sensitivity, *, motion_density=None, raw_atom_indices=None):
        sensitivity = torch.tensor(
            sensitivity,
            dtype=self.density.dtype,
            device=self.density.device,
        )
        expected_shape = (
            len(self.descriptor.descriptor_atom_indices),
            sum(self.descriptor.shell_sizes),
        )
        if sensitivity.shape != expected_shape:
            raise ValueError("sensitivity does not match the descriptor layout")
        motion_density = (
            self.density
            if motion_density is None
            else self.descriptor.as_ao_density_tensor(motion_density)
        )
        coordinate_jacobians = self._coordinate_jacobians(
            motion_density,
            raw_atom_indices,
        )
        gradient_parts = []
        potential = torch.zeros_like(self.density)
        for overlap, coordinate, jacobian, shell_sensitivity in zip(
            self.descriptor.overlap_shells,
            coordinate_jacobians,
            self.eigenvalue_jacobians,
            sensitivity.split(self.descriptor.shell_sizes, dim=-1),
            strict=True,
        ):
            block_adjoint = torch.einsum("av,avpq->apq", shell_sensitivity, jacobian)
            potential += torch.einsum(
                "rap,apq,saq->rs", overlap, block_adjoint, overlap
            )
            gradient_parts.append(
                torch.einsum("bxapq,apq->bx", coordinate, block_adjoint)
            )
        gradient = torch.stack(gradient_parts, dim=0).sum(dim=0)
        return (
            immutable_array(gradient.detach().cpu().numpy(), dtype=np.float64),
            immutable_array(potential.detach().cpu().numpy(), dtype=np.float64),
        )


__all__ = ["DescriptorDerivativeWorkspace"]
