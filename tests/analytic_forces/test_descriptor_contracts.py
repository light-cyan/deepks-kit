import numpy as np
import pytest
import torch

from deepks.descriptor import (
    DescriptorDifferentiabilityError,
    occupied_virtual_gradient,
    validate_differentiability,
)


def test_nondegenerate_descriptor_block_is_accepted():
    diagnostics = validate_differentiability(
        values=np.array([[0.2, 0.8]]),
        shell_sizes=(2,),
        n_occupied=2,
        sensitivity=np.array([[0.3, -0.7]]),
    )

    assert diagnostics.structural_zero_blocks == ()
    assert diagnostics.minimum_scaled_gap > 1.0


def test_structural_zero_block_with_equal_sensitivity_is_accepted():
    diagnostics = validate_differentiability(
        values=np.array([[0.0, 0.0, 0.7]]),
        shell_sizes=(3,),
        n_occupied=1,
        sensitivity=np.array([[0.4, 0.4, -0.2]]),
    )

    assert diagnostics.structural_zero_blocks == ((0, 0, 0, 2),)


def test_structural_zero_block_with_unequal_sensitivity_is_rejected():
    with pytest.raises(DescriptorDifferentiabilityError) as error:
        validate_differentiability(
            values=np.array([[0.0, 0.0, 0.7]]),
            shell_sizes=(3,),
            n_occupied=1,
            sensitivity=np.array([[0.4, 0.5, -0.2]]),
        )

    message = str(error.value)
    assert "descriptor atom 0, shell 0" in message
    assert "structural zero block sensitivity spread" in message
    assert "exceeds" in message


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array([[0.5, 0.5]]), id="symmetry-splitting"),
        pytest.param(np.array([[0.5, 0.50000005]]), id="near-crossing"),
    ],
)
def test_splitting_and_near_crossing_blocks_are_rejected_with_context(
    values,
):
    with pytest.raises(DescriptorDifferentiabilityError) as error:
        validate_differentiability(
            values=values,
            shell_sizes=(2,),
            n_occupied=2,
            sensitivity=np.array([[0.2, 0.2]]),
        )

    message = str(error.value)
    assert "descriptor atom 0, shell 0" in message
    assert "eigenvalue gap" in message
    assert "block positions 0:2" in message
    assert "does not exceed" in message


def test_occupied_virtual_gradient_matches_the_explicit_ao_formula():
    ao_operators = torch.tensor(
        [
            [
                [0.7, -0.2, 0.1, 0.3],
                [0.4, 0.5, -0.6, 0.2],
                [-0.1, 0.8, 0.9, -0.4],
                [0.3, -0.7, 0.2, 0.6],
            ],
            [
                [-0.5, 0.4, 0.3, -0.2],
                [0.1, 0.8, -0.7, 0.6],
                [0.2, -0.3, 0.5, 0.9],
                [-0.4, 0.7, 0.1, -0.8],
            ],
        ],
        dtype=torch.float64,
    )
    occupied_coefficients = torch.tensor(
        [
            [0.8, -0.1],
            [0.3, 0.7],
            [-0.4, 0.2],
            [0.1, -0.6],
        ],
        dtype=torch.float64,
    )
    virtual_coefficients = torch.tensor(
        [
            [0.2, -0.5],
            [-0.7, 0.1],
            [0.6, 0.4],
            [0.3, -0.8],
        ],
        dtype=torch.float64,
    )
    occupations = torch.tensor([2.0, 0.5], dtype=torch.float64)

    actual = occupied_virtual_gradient(
        ao_operators,
        virtual_coefficients,
        occupied_coefficients,
        occupations,
    )
    expected_matrices = torch.stack(
        [
            virtual_coefficients.T
            @ ao_operator
            @ (occupied_coefficients * occupations)
            for ao_operator in ao_operators
        ]
    )

    assert actual.shape == (2, 4)
    torch.testing.assert_close(
        actual,
        expected_matrices.flatten(-2),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    unit_occupation = occupied_virtual_gradient(
        ao_operators,
        virtual_coefficients,
        occupied_coefficients,
        torch.ones_like(occupations),
    ).reshape(2, 2, 2)
    torch.testing.assert_close(
        actual.reshape(2, 2, 2),
        unit_occupation * occupations,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
