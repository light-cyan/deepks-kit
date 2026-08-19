"""Method-independent orbital-coordinate transformations."""

import torch


def occupied_virtual_gradient(
    ao_operators: torch.Tensor,
    virtual_coefficients: torch.Tensor,
    occupied_coefficients: torch.Tensor,
    occupations: torch.Tensor,
) -> torch.Tensor:
    """Transform AO operators to flattened occupied-virtual gradients."""
    gradient = torch.einsum(
        "pa,qi,...pq->...ai",
        virtual_coefficients,
        occupied_coefficients * occupations,
        ao_operators,
    )
    return gradient.flatten(-2)
