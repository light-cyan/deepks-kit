from copy import deepcopy

import numpy as np
import pytest
import torch

from deepks.deephf import (
    DeePHFCapabilityError,
    RHFDeePHFGradients,
    RHFDeePHFZVectorGradients,
    RKSAdjointAdapter,
    RKSAdjointError,
    RKSDeePHF,
    RKSDeePHFGradients,
    RKSDeePHFZVectorGradients,
    RKSResponseAdapter,
    generate_rhf_force_frame,
)
from deepks.deephf.pyscf_rks import rks_adjoint_integrity_fingerprint
from deepks.descriptor import DescriptorDifferentiabilityError
from deepks.model.model import CorrNet


DRIVER_RESULT_FIELDS = (
    "adjoint_result",
    "descriptor_diagnostics",
    "reference_gradient",
    "dq_dR_explicit",
    "correction_gradient_explicit",
    "correction_gradient_metric",
    "correction_gradient_adjoint_fixed_grid",
    "correction_gradient_adjoint_grid_coordinate",
    "correction_gradient_adjoint_grid_weight",
    "correction_gradient_adjoint_nuclear",
    "correction_gradient_adjoint_metric",
    "correction_gradient_grid_coordinate",
    "correction_gradient_grid_weight",
    "correction_gradient_occupied_virtual",
    "correction_gradient_response",
    "correction_gradient",
    "de_full",
    "de",
)


def _assert_driver_cleared(driver):
    assert all(getattr(driver, name, None) is None for name in DRIVER_RESULT_FIELDS)


def _immutable(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@pytest.mark.parametrize("backend", [None, "", "Z", "rks_zvector", True])
def test_unknown_rks_gradient_backend_is_rejected(rks_oracle_case, backend):
    with pytest.raises(
        ValueError,
        match="RKS gradient backend must be 'direct' or 'zvector'",
    ):
        rks_oracle_case.method.nuc_grad_method(backend=backend)


@pytest.mark.parametrize(
    ("backend", "options", "match"),
    [
        pytest.param(
            "zvector",
            {"cphf_tolerance": 1.0e-12},
            "unsupported zvector backend options: cphf_tolerance",
            id="direct-option-in-zvector",
        ),
        pytest.param(
            "direct",
            {"objective_symmetry_tolerance": 1.0e-12},
            "unsupported direct backend options: objective_symmetry_tolerance",
            id="adjoint-option-in-direct",
        ),
        pytest.param(
            "zvector",
            {"fallback": "direct"},
            "unsupported zvector backend options: fallback",
            id="fallback",
        ),
    ],
)
def test_rks_backend_options_are_strictly_namespaced(
    rks_oracle_case,
    backend,
    options,
    match,
):
    with pytest.raises(ValueError, match=match):
        rks_oracle_case.method.nuc_grad_method(
            backend=backend,
            **options,
        )


def test_rks_direct_remains_default_and_adjoint_options_are_independent(
    rks_oracle_case,
    monkeypatch,
):
    method = rks_oracle_case.method
    monkeypatch.setitem(method.response_options, "cphf_tolerance", 1.0e-11)
    monkeypatch.setitem(
        method.adjoint_options,
        "objective_symmetry_tolerance",
        1.0e-10,
    )

    direct = method.nuc_grad_method()
    zvector = method.nuc_grad_method(
        backend="zvector",
        invariant_tolerance=2.0e-9,
    )

    assert type(direct) is RKSDeePHFGradients
    assert direct.backend == "direct"
    assert type(zvector) is RKSDeePHFZVectorGradients
    assert zvector.backend == "zvector"
    assert dict(zvector.adjoint_options) == {"invariant_tolerance": 2.0e-9}
    with pytest.raises(TypeError):
        zvector.adjoint_options["residual_tolerance"] = 1.0e-8
    with pytest.raises(TypeError, match="adjoint_options must be a mapping"):
        RKSDeePHF(
            rks_oracle_case.reference,
            deepcopy(rks_oracle_case.model),
            projector_basis=rks_oracle_case.model._pbas,
            adjoint_options=[("residual_tolerance", 1.0e-9)],
        )


def test_rks_zvector_driver_public_bindings_are_read_only(rks_oracle_case):
    driver = rks_oracle_case.method.nuc_grad_method(backend="zvector")
    for name, value in (
        ("base", object()),
        ("mol", object()),
        ("backend", "direct"),
        ("adjoint_options", {}),
    ):
        with pytest.raises(AttributeError):
            setattr(driver, name, value)


@pytest.mark.parametrize("corruption", ["base", "mol", "backend", "options"])
def test_rks_zvector_corrupted_binding_fails_and_clears_every_result(
    rks_oracle_case,
    corruption,
):
    driver = rks_oracle_case.method.nuc_grad_method(backend="zvector")
    for name in DRIVER_RESULT_FIELDS:
        setattr(driver, name, object())
    if corruption == "base":
        driver._base = object()
    elif corruption == "mol":
        driver._mol = object()
    elif corruption == "backend":
        driver._backend = "direct"
    else:
        driver._adjoint_options = {}

    with pytest.raises(RKSAdjointError, match="driver binding is invalid"):
        driver.kernel()
    _assert_driver_cleared(driver)




def test_rks_zvector_state_failures_never_enter_adjoint_or_direct_fallback(
    rks_oracle_case,
    monkeypatch,
):
    method = rks_oracle_case.method
    driver = method.nuc_grad_method(backend="zvector")
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("an invalid RKS state entered a response backend")

    monkeypatch.setattr(RKSAdjointAdapter, "solve", forbidden)
    monkeypatch.setattr(RKSResponseAdapter, "solve", forbidden)
    monkeypatch.setattr(RKSDeePHFGradients, "kernel", forbidden)
    original_weight = method.model.linear.weight[0, 0].detach().clone()
    with torch.no_grad():
        method.model.linear.weight[0, 0] = torch.nan
    try:
        with pytest.raises(
            DeePHFCapabilityError,
            match="parameters and buffers must be finite",
        ):
            driver.kernel()
        _assert_driver_cleared(driver)
    finally:
        with torch.no_grad():
            method.model.linear.weight[0, 0] = original_weight

    original_descriptor = method._descriptor
    method._descriptor = object()
    try:
        with pytest.raises(
            DeePHFCapabilityError,
            match="descriptor identity changed",
        ):
            driver.kernel()
        _assert_driver_cleared(driver)
    finally:
        method._descriptor = original_descriptor

    original_converged = method.reference.converged
    method.reference.converged = False
    try:
        with pytest.raises(
            DeePHFCapabilityError,
            match="scientific state changed|must be converged",
        ):
            driver.kernel()
        _assert_driver_cleared(driver)
    finally:
        method.reference.converged = original_converged
    assert calls == []


def test_rks_zvector_nondifferentiable_descriptor_fails_before_adjoint(
    rks_oracle_case,
    monkeypatch,
):
    projector_basis = [[4, [0.2, 1.0]]]
    model = CorrNet(
        input_dim=9,
        hidden_sizes=(2,),
        proj_basis=projector_basis,
    ).double()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(0.007)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    method = RKSDeePHF(
        rks_oracle_case.reference,
        model.eval(),
        projector_basis=projector_basis,
    )
    driver = method.nuc_grad_method(backend="zvector")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a nondifferentiable descriptor entered an adjoint solve")

    monkeypatch.setattr(RKSAdjointAdapter, "solve", forbidden)
    with pytest.raises(
        DescriptorDifferentiabilityError,
        match="eigenvalue gap|structural zero block",
    ):
        driver.kernel()
    _assert_driver_cleared(driver)


def test_rks_zvector_rejects_scanner_rhf_drivers_and_force_data(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    driver = method.nuc_grad_method(backend="zvector")

    with pytest.raises(RKSAdjointError, match="gradient scanner"):
        driver.as_scanner()
    with pytest.raises(TypeError, match="requires an exact DeePHF method"):
        RHFDeePHFGradients(method)
    with pytest.raises(TypeError, match="requires an exact DeePHF method"):
        RHFDeePHFZVectorGradients(method)
    with pytest.raises(DeePHFCapabilityError, match="native pyscf.scf.hf.RHF"):
        generate_rhf_force_frame(
            rks_oracle_case.reference,
            projector_basis=rks_oracle_case.model._pbas,
            e_target=np.float64(rks_oracle_case.reference.e_tot),
            f_target=np.zeros((rks_oracle_case.reference.mol.natm, 3)),
        )
    assert not hasattr(driver, "dq_dR_response")
    assert not hasattr(driver, "dq_dR_relaxed")
