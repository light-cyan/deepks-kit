import ast
import inspect
import textwrap

import deepks.deephf.gradient as direct_module
import deepks.deephf.zvector as adjoint_module

from deepks.deephf.gradient import RHFDeePHFGradients
from deepks.deephf.gradient import (
    RKSDeePHFGradients,
    UHFDeePHFGradients,
    UKSDeePHFGradients,
)
from deepks.deephf.zvector import (
    RKSDeePHFZVectorGradients,
    UHFDeePHFZVectorGradients,
    UKSDeePHFZVectorGradients,
)
from deepks.deephf.zvector import RHFDeePHFZVectorGradients


def called_attributes(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }


def test_direct_drivers_do_not_call_adjoint_entry_points():
    for driver in (
        RHFDeePHFGradients,
        UHFDeePHFGradients,
        RKSDeePHFGradients,
        UKSDeePHFGradients,
    ):
        calls = called_attributes(driver._compact_kernel) | called_attributes(driver._detail_kernel)
        assert "_zvector_inputs" not in calls
        assert "adjoint" not in calls


def test_zvector_drivers_do_not_call_direct_response_entry_points():
    for driver in (
        RHFDeePHFZVectorGradients,
        UHFDeePHFZVectorGradients,
        RKSDeePHFZVectorGradients,
        UKSDeePHFZVectorGradients,
    ):
        calls = called_attributes(driver._compact_kernel) | called_attributes(driver._detail_kernel)
        assert "_solve_response" not in calls
        assert "response" not in calls


def test_direct_and_scalar_adjoint_drivers_are_physically_separate():
    assert direct_module.__file__ != adjoint_module.__file__
    assert all(driver.__module__ == direct_module.__name__ for driver in (
        RHFDeePHFGradients,
        RKSDeePHFGradients,
        UHFDeePHFGradients,
        UKSDeePHFGradients,
    ))
    assert all(driver.__module__ == adjoint_module.__name__ for driver in (
        RHFDeePHFZVectorGradients,
        RKSDeePHFZVectorGradients,
        UHFDeePHFZVectorGradients,
        UKSDeePHFZVectorGradients,
    ))
