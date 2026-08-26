from copy import deepcopy

import numpy as np
import pytest
import torch
from pyscf import gto

import deepks.model.model as model_module
from deepks.deephf import DeePHF
from deepks.deephf.capabilities import DeePHFCapabilityError
from deepks.model.model import (
    CorrNet,
    model_execution_state_evidence,
    model_execution_state_fingerprint,
)
from tests.reference_utils import build_cpu_reference as build_reference


PROJECTOR_BASIS = [[0, [0.8, 1.0]]]
THERMAL_PROJECTOR_BASIS = [[1, [0.8, 1.0]]]


def _model(
    *,
    embedding=None,
    elem_table=None,
    input_dim=1,
    projector_basis=PROJECTOR_BASIS,
):
    model = CorrNet(
        input_dim=input_dim,
        hidden_sizes=(1,),
        actv_fn="gelu",
        proj_basis=projector_basis,
        embedding=embedding,
        elem_table=elem_table,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.fill_(0.31)
        model.linear.weight.fill_(0.23)
        model.linear.bias.fill_(0.17)
    return model.eval()


def _thermal_method():
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    return DeePHF(
        build_reference(molecule, "rhf"),
        _model(
            embedding={"type": "thermal"},
            input_dim=3,
            projector_basis=THERMAL_PROJECTOR_BASIS,
        ),
        projector_basis=THERMAL_PROJECTOR_BASIS,
    )


@pytest.fixture
def method():
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    return DeePHF(
        build_reference(molecule, "rhf"),
        _model(),
        projector_basis=PROJECTOR_BASIS,
    )


def _assert_reuse_rejected(method, mutation, *, publish_gradient=False):
    driver = (
        method.nuc_grad_method(backend="direct", retain_details=False)
        if publish_gradient
        else None
    )
    boundary_rejected = False
    with pytest.raises(DeePHFCapabilityError, match="scientific state changed"):
        with method.calculation():
            method.kernel()
            if driver is not None:
                driver.kernel()
            mutation(method.model)
            try:
                method.correction_energy()
            except DeePHFCapabilityError:
                boundary_rejected = True
                raise
    assert boundary_rejected
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)
    if driver is not None:
        assert driver.de is None
    assert method.operation_counts["cache_invalidations"] == 1
    if driver is None:
        assert method.operation_counts.get("public_model_value_fingerprints", 0) == 0


def _mutate_bias(model):
    with torch.no_grad():
        model.linear.bias.add_(0.2)


def _assert_untracked_value_mutation_rejected(method, mutation):
    old_energy = method.kernel()
    old_gradient = method.gradient(backend="direct")
    driver = method.nuc_grad_method(backend="direct", retain_details=False)
    boundary_rejected = False
    with pytest.raises(DeePHFCapabilityError, match="scientific state changed"):
        with method.calculation():
            method.kernel()
            driver.kernel()
            mutation(method.model)
            try:
                method.correction_energy()
            except DeePHFCapabilityError:
                boundary_rejected = True
                raise
    assert boundary_rejected
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)
    assert driver.de is None
    assert method.operation_counts["cache_invalidations"] == 1
    assert method.operation_counts["public_model_value_fingerprints"] == 2

    new_energy = method.kernel()
    new_gradient = method.gradient(backend="direct")
    assert abs(new_energy - old_energy) > 1.0e-6
    assert np.max(np.abs(new_gradient - old_gradient)) > 1.0e-8


def test_data_parameter_mutation_fails_at_public_reuse_boundary(method):
    _assert_untracked_value_mutation_rejected(
        method,
        lambda model: model.linear.weight.data.add_(0.4),
    )


def test_data_buffer_mutation_fails_at_public_reuse_boundary():
    method = _thermal_method()

    def mutate_running_variance(model):
        model.embedder.running_var.data.fill_(4.0)

    _assert_untracked_value_mutation_rejected(method, mutate_running_variance)


def _replacement_helper(name, original):
    if name == "pad_masked":
        def replacement(tensor, mask, padding_value=0):
            return original(tensor, mask, padding_value) + 0.4 * mask.to(tensor)
    elif name == "masked_softmax":
        def replacement(input, mask, dim=-1):
            return original(input, mask, dim) * 0.5
    else:
        def replacement(padded, mask):
            return original(padded, mask) + 0.4
    return replacement


@pytest.mark.parametrize(
    "helper_name",
    ("pad_masked", "masked_softmax", "unpad_masked"),
)
def test_thermal_helper_replacement_changes_fingerprint_and_rejects_cached_energy(
    helper_name,
):
    method = _thermal_method()
    original = getattr(model_module, helper_name)
    replacement = _replacement_helper(helper_name, original)
    original_fingerprint = model_execution_state_fingerprint(method.model)
    original_energy = method.correction_energy()
    try:
        _assert_reuse_rejected(
            method,
            lambda _model: setattr(model_module, helper_name, replacement),
            publish_gradient=True,
        )
        changed_fingerprint = model_execution_state_fingerprint(method.model)
        assert changed_fingerprint != original_fingerprint
        fresh_energy = method.correction_energy()
        assert abs(fresh_energy - original_energy) > 1.0e-6
    finally:
        setattr(model_module, helper_name, original)
    assert model_execution_state_fingerprint(method.model) == original_fingerprint


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model: setattr(model.densenet, "actv_fn", torch.tanh),
        lambda model: setattr(model.densenet, "use_resnet", not model.densenet.use_resnet),
        lambda model: setattr(
            model.densenet,
            "dts",
            torch.nn.ParameterList(
                torch.nn.Parameter(torch.ones(layer.out_features, dtype=torch.float64))
                for layer in model.densenet.layers
            ),
        ),
        lambda model: model.register_forward_hook(lambda _module, _args, output: output + 0.5),
        lambda model: setattr(model, "_compiled_call_impl", lambda *args: model.forward(*args[1:])),
        lambda model: model.train(),
        lambda model: setattr(model, "input_dim", model.input_dim + 1),
        lambda model: model._pbas[0][1].__setitem__(0, model._pbas[0][1][0] + 0.1),
        lambda model: setattr(
            model.linear,
            "weight",
            torch.nn.Parameter(model.linear.weight.detach().clone()),
        ),
    ),
    ids=(
        "activation",
        "residual-policy",
        "residual-scaling",
        "hook",
        "compiled-dispatch",
        "mode",
        "input-metadata",
        "projector-metadata",
        "parameter-replacement",
    ),
)
def test_corrnet_graph_mutations_fail_before_cached_energy_reuse(method, mutation):
    _assert_reuse_rejected(method, mutation)


def test_embedder_and_element_configuration_mutations_are_evidence(method):
    embedded = _model(embedding="trace")
    initial = model_execution_state_evidence(embedded)
    embedded.embedder.shell_sec = (1,)
    assert model_execution_state_evidence(embedded) != initial

    element_model = _model(elem_table=([1], [0.5]))
    initial = model_execution_state_evidence(element_model)
    element_model.elem_dict[1] = 0.75
    assert model_execution_state_evidence(element_model) != initial


def test_buffer_replacement_fails_before_cached_reuse(method):
    embedded = DeePHF(
        method.reference,
        _model(embedding={"type": "thermal"}),
        projector_basis=PROJECTOR_BASIS,
    )
    _assert_reuse_rejected(
        embedded,
        lambda model: setattr(
            model.embedder,
            "running_mean",
            model.embedder.running_mean.detach().clone(),
        ),
    )


def test_global_hook_and_trusted_implementation_mutations_fail_closed(method, monkeypatch):
    handle = None

    def add_global_hook(_model):
        nonlocal handle
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda _module, _args, output: output
        )

    try:
        _assert_reuse_rejected(method, add_global_hook)
    finally:
        if handle is not None:
            handle.remove()

    replacement = DeePHF(
        method.reference,
        _model(),
        projector_basis=PROJECTOR_BASIS,
    )
    original = CorrNet.forward

    def changed_forward(self, values):
        return original(self, values) + 0.1

    _assert_reuse_rejected(
        replacement,
        lambda _model: monkeypatch.setattr(CorrNet, "forward", changed_forward),
    )


@pytest.mark.parametrize(
    "setter",
    (
        lambda model: model.set_normalization(shift=[0.4], scale=[1.7]),
        lambda model: model.set_prefitting(weight=[0.8], bias=[-0.2], trainable=True),
        lambda model: model.set_energy_const(0.9),
    ),
    ids=("normalization", "prefitting", "energy-constant"),
)
def test_builtin_setters_preserve_storage_contract_and_reject_mid_transaction_reuse(
    method,
    setter,
):
    parameters = tuple(method.model.parameters())
    identities = tuple(id(parameter) for parameter in parameters)
    versions = tuple(parameter._version for parameter in parameters)
    dtypes = tuple(parameter.dtype for parameter in parameters)
    devices = tuple(parameter.device for parameter in parameters)

    _assert_reuse_rejected(method, setter)

    updated = tuple(method.model.parameters())
    assert tuple(id(parameter) for parameter in updated) == identities
    assert tuple(parameter.dtype for parameter in updated) == dtypes
    assert tuple(parameter.device for parameter in updated) == devices
    assert any(
        parameter._version > version
        for parameter, version in zip(updated, versions, strict=True)
    )


def test_prefitting_setter_preserves_declared_trainability():
    model = _model()
    model.set_prefitting([0.2], [0.3], trainable=False)
    assert model.linear.weight.requires_grad is False
    assert model.linear.bias.requires_grad is False
    model.set_prefitting([0.4], [0.5], trainable=True)
    assert model.linear.weight.requires_grad is True
    assert model.linear.bias.requires_grad is True


def test_setter_between_independent_calculations_uses_new_values(method):
    first = method.kernel()
    method.model.set_energy_const(0.75)
    second = method.kernel()
    assert second == pytest.approx(first + 0.75, abs=1.0e-12)


def test_nonzero_activation_change_rejects_stale_energy_and_gradient(method):
    old_energy = method.kernel()
    old_gradient = method.gradient(backend="direct")

    with pytest.raises(DeePHFCapabilityError, match="scientific state changed"):
        with method.calculation():
            method.kernel()
            method.model.densenet.actv_fn = torch.tanh
            method.gradient(backend="direct")

    new_energy = method.kernel()
    new_gradient = method.gradient(backend="direct")
    assert abs(new_energy - old_energy) > 1.0e-6
    assert np.max(np.abs(new_gradient - old_gradient)) > 1.0e-8


def test_generic_model_output_is_recomputed_at_each_interruptible_boundary(method):
    class PythonScale(torch.nn.Module):
        input_dim = 1

        def __init__(self):
            super().__init__()
            self.scale = 0.25
            self.calls = 0

        def forward(self, values):
            self.calls += 1
            return values.sum() * self.scale

    model = PythonScale()
    generic = DeePHF(
        method.reference,
        model,
        projector_basis=deepcopy(PROJECTOR_BASIS),
    )
    with generic.calculation():
        first = generic.correction_energy()
        model.scale = 0.5
        second = generic.correction_energy()
    assert model.calls == 2
    assert second == pytest.approx(2.0 * first, abs=1.0e-12)
    assert generic.operation_counts["conservative_model_fingerprints"] == 1
    assert generic.operation_counts["generic_model_cache_resets"] == 1


def test_no_grad_tensor_mutation_and_storage_replacement_fail_closed(method):
    _assert_reuse_rejected(
        method,
        _mutate_bias,
    )

    replacement_method = DeePHF(
        method.reference,
        _model(),
        projector_basis=PROJECTOR_BASIS,
    )
    _assert_reuse_rejected(
        replacement_method,
        lambda model: setattr(
            model.linear,
            "bias",
            torch.nn.Parameter(model.linear.bias.detach().clone()),
        ),
    )
