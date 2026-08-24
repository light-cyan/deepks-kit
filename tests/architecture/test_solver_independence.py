import ast
import inspect
import textwrap

from deepks.deephf.gradient import RHFDeePHFGradients
from deepks.deephf.rks_gradient import RKSDeePHFGradients
from deepks.deephf.rks_zvector import RKSDeePHFZVectorGradients
from deepks.deephf.uhf_gradient import UHFDeePHFGradients
from deepks.deephf.uhf_zvector import UHFDeePHFZVectorGradients
from deepks.deephf.uks_gradient import UKSDeePHFGradients
from deepks.deephf.uks_zvector import UKSDeePHFZVectorGradients
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
