from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from pyscf import gto, scf
from torch.nn.modules import module as torch_module

from deepks.deephf import DeePHF, DeePHFCapabilityError
from deepks.deephf.scanner import (
    RHFDeePHFGradientScanner,
    RHFDeePHFScannerError,
)
from deepks.deephf.pyscf_rhf import (
    RHFScannerReferenceError,
    molecule_science_fingerprint,
    reference_fingerprint,
)
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]]]
REFERENCE_COORDINATES = np.array(
    [
        [0.03, -0.08, 0.04],
        [0.21, 0.17, 1.41],
    ],
    dtype=np.float64,
)


def _molecule(coordinates=REFERENCE_COORDINATES, *, basis="sto-3g"):
    return gto.M(
        atom=list(zip(("H", "H"), np.asarray(coordinates))),
        basis=basis,
        unit="Bohr",
        symmetry=False,
        cart=False,
        verbose=0,
    )


def _fresh_reference(coordinates=REFERENCE_COORDINATES):
    reference = scf.RHF(_molecule(coordinates))
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.kernel(dm0=None)
    assert reference.converged
    return reference


def _model():
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=PROJECTOR_BASIS,
        input_shift=[0.13],
        input_scale=[0.81],
        output_scale=1.17,
    ).double()
    with torch.no_grad():
        model.linear.weight.fill_(0.037)
        model.linear.bias.fill_(0.011)
        first_layer, output_layer = model.densenet.layers
        first_layer.weight[:] = torch.tensor(
            [[0.29], [-0.17]],
            dtype=torch.float64,
        )
        first_layer.bias[:] = torch.tensor(
            [0.03, -0.02],
            dtype=torch.float64,
        )
        output_layer.weight[:] = torch.tensor(
            [[0.21, -0.14]],
            dtype=torch.float64,
        )
        output_layer.bias.fill_(0.019)
        model.energy_const.fill_(0.007)
    return model.eval()


@dataclass(frozen=True)
class _ScannerCase:
    model: torch.nn.Module
    method: DeePHF
    driver: object
    scanner: RHFDeePHFGradientScanner


@pytest.fixture
def scanner_case():
    model = _model()
    method = DeePHF(
        _fresh_reference(),
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    driver = method.nuc_grad_method(backend="zvector")
    scanner = driver.as_scanner()
    return _ScannerCase(
        model=model,
        method=method,
        driver=driver,
        scanner=scanner,
    )


def _fresh_deephf_result(coordinates, model):
    method = DeePHF(
        _fresh_reference(coordinates),
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    energy = method.kernel()
    gradient = method.nuc_grad_method(backend="zvector").kernel()
    return energy, gradient


def _fresh_direct_result(coordinates, model):
    method = DeePHF(
        _fresh_reference(coordinates),
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    energy = method.kernel()
    gradient = method.nuc_grad_method(backend="direct").kernel()
    return energy, gradient


def _assert_no_current_result(scanner):
    assert not scanner.converged
    for name in (
        "mol",
        "reference",
        "method",
        "gradient_driver",
        "e_tot",
        "de",
        "model_state_fingerprint",
    ):
        assert getattr(scanner, name) is None


def test_scanner_rebuilds_fresh_objects_for_a_b_a_and_matches_fresh_methods(
    scanner_case,
):
    scanner = scanner_case.scanner
    with pytest.raises(AttributeError):
        scanner.backend = "direct"
    with pytest.raises(TypeError):
        scanner.response_options["residual_tolerance"] = 1.0
    scanner_case.driver.response_options["residual_tolerance"] = -1.0
    assert dict(scanner.response_options) == {}
    displaced = REFERENCE_COORDINATES.copy()
    displaced[1] += np.array([0.04, -0.03, 0.05])
    geometries = (REFERENCE_COORDINATES, displaced, REFERENCE_COORDINATES)
    object_graphs = []

    for coordinates in geometries:
        scanned_energy, scanned_gradient = scanner(coordinates)
        fresh_energy, fresh_gradient = _fresh_deephf_result(
            coordinates,
            scanner_case.model,
        )

        np.testing.assert_allclose(
            scanned_energy,
            fresh_energy,
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            scanned_gradient,
            fresh_gradient,
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        assert scanner.base is scanner_case.driver
        assert scanner.backend == "zvector"
        assert scanner.gradient_driver.backend == "zvector"
        assert scanner.gradient_driver.base is scanner.method
        assert scanner.reference is scanner.method.reference
        assert scanner.mol is scanner.reference.mol
        assert scanner.converged
        assert not scanner.de.flags.writeable
        object_graphs.append(
            (
                scanner.mol,
                scanner.reference,
                scanner.method,
                scanner.gradient_driver,
            )
        )

        internal_gradient = scanner.de.copy()
        scanned_gradient[0, 0] += 1.0
        np.testing.assert_array_equal(scanner.de, internal_gradient)

    assert all(
        len({id(objects[index]) for objects in object_graphs}) == 3
        for index in range(4)
    )


def test_scanner_preserves_an_explicit_direct_backend(scanner_case):
    direct_driver = scanner_case.method.nuc_grad_method(backend="direct")
    scanner = direct_driver.as_scanner()

    energy, gradient = scanner(REFERENCE_COORDINATES)
    fresh_energy, fresh_gradient = _fresh_direct_result(
        REFERENCE_COORDINATES,
        scanner_case.model,
    )

    assert scanner.base is direct_driver
    assert scanner.backend == "direct"
    assert scanner.gradient_driver.backend == "direct"
    np.testing.assert_allclose(energy, fresh_energy, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        gradient,
        fresh_gradient,
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_scanner_snapshots_independent_method_and_driver_option_namespaces():
    method_response_options = {"cphf_tolerance": 1.0e-11}
    method_adjoint_options = {"objective_symmetry_tolerance": 1.0e-10}
    driver_options = {"max_cycle": 64}
    method = DeePHF(
        _fresh_reference(),
        _model(),
        projector_basis=PROJECTOR_BASIS,
        response_options=method_response_options,
        adjoint_options=method_adjoint_options,
    )
    driver = method.nuc_grad_method(backend="zvector", **driver_options)
    scanner = driver.as_scanner()

    method.response_options["objective_symmetry_tolerance"] = 1.0e-10
    method.adjoint_options["cphf_tolerance"] = 1.0e-11
    driver.response_options["cphf_tolerance"] = 1.0e-11

    energy, gradient = scanner(REFERENCE_COORDINATES)

    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    assert scanner.backend == "zvector"
    assert dict(scanner.response_options) == driver_options
    assert scanner.method.response_options == method_response_options
    assert scanner.method.adjoint_options == method_adjoint_options
    assert scanner.gradient_driver.response_options == driver_options


def test_scanner_does_not_mutate_the_original_reference_model_or_input_molecule(
    scanner_case,
):
    scanner = scanner_case.scanner
    original_reference = scanner_case.method.reference
    reference_state = reference_fingerprint(original_reference)
    original_method_state = (
        scanner_case.method.e_base,
        scanner_case.method.e_corr,
        scanner_case.method.e_tot,
    )
    model_state = {
        name: value.detach().clone()
        for name, value in scanner_case.model.state_dict().items()
    }
    model_metadata = deepcopy(scanner_case.model._pbas)
    model_training = scanner_case.model.training
    model_gradients = {
        name: parameter.grad
        for name, parameter in scanner_case.model.named_parameters()
    }
    displaced = REFERENCE_COORDINATES.copy()
    displaced[1] += np.array([-0.02, 0.04, 0.03])
    input_molecule = _molecule(displaced)
    input_coordinates = input_molecule.atom_coords(unit="Bohr").copy()

    scanner(input_molecule)
    scanner(REFERENCE_COORDINATES)

    assert reference_fingerprint(original_reference) == reference_state
    assert (
        scanner_case.method.e_base,
        scanner_case.method.e_corr,
        scanner_case.method.e_tot,
    ) == original_method_state
    assert scanner_case.model.training is model_training
    assert scanner_case.model._pbas == model_metadata
    for name, value in scanner_case.model.state_dict().items():
        torch.testing.assert_close(value, model_state[name], rtol=0.0, atol=0.0)
    for name, parameter in scanner_case.model.named_parameters():
        assert parameter.grad is model_gradients[name]
    np.testing.assert_array_equal(
        input_molecule.atom_coords(unit="Bohr"),
        input_coordinates,
    )


def test_invalid_atom_selection_fails_before_fresh_scf_and_valid_selection_works(
    scanner_case,
    monkeypatch,
):
    scanner = scanner_case.scanner
    anchor_before = scanner._root_anchor
    scf_calls = []

    def unexpected_kernel(*args, **kwargs):
        scf_calls.append((args, kwargs))
        raise AssertionError("fresh SCF must not run for an invalid atmlst")

    with monkeypatch.context() as patch:
        patch.setattr(scf.hf.RHF, "kernel", unexpected_kernel, raising=False)
        with pytest.raises(TypeError, match="atom indices must be integers"):
            scanner(REFERENCE_COORDINATES, atmlst=(True,))

    assert scf_calls == []
    assert scanner._root_anchor is anchor_before
    _assert_no_current_result(scanner)

    energy, selected_gradient = scanner(
        REFERENCE_COORDINATES,
        atmlst=(1,),
    )
    fresh_energy, fresh_gradient = _fresh_deephf_result(
        REFERENCE_COORDINATES,
        scanner_case.model,
    )
    np.testing.assert_allclose(energy, fresh_energy, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        selected_gradient,
        fresh_gradient[[1]],
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    assert scanner.de.shape == (1, 3)


@pytest.mark.parametrize(
    "register_name",
    (
        "register_forward_pre_hook",
        "register_forward_hook",
        "register_full_backward_pre_hook",
        "register_full_backward_hook",
    ),
)
def test_scanner_rejects_local_module_execution_hooks_before_fresh_scf(
    scanner_case,
    monkeypatch,
    register_name,
):
    scanner = scanner_case.scanner
    anchor_before = scanner._root_anchor
    scf_calls = []

    def unexpected_kernel(*args, **kwargs):
        scf_calls.append((args, kwargs))
        raise AssertionError("fresh SCF must not run with module execution hooks")

    register = getattr(scanner_case.model.densenet, register_name)
    hook = register(lambda *_args, **_kwargs: None)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(scf.hf.RHF, "kernel", unexpected_kernel, raising=False)
            with pytest.raises(
                RHFDeePHFScannerError,
                match="cannot contain module execution hooks",
            ):
                scanner(REFERENCE_COORDINATES)
    finally:
        hook.remove()

    assert scf_calls == []
    _assert_no_current_result(scanner)
    assert scanner._root_anchor is anchor_before


@pytest.mark.parametrize(
    "register_name",
    (
        "register_module_forward_pre_hook",
        "register_module_forward_hook",
        "register_module_full_backward_pre_hook",
        "register_module_full_backward_hook",
    ),
)
def test_scanner_rejects_global_module_execution_hooks_before_fresh_scf(
    scanner_case,
    monkeypatch,
    register_name,
):
    scanner = scanner_case.scanner
    anchor_before = scanner._root_anchor
    scf_calls = []

    def unexpected_kernel(*args, **kwargs):
        scf_calls.append((args, kwargs))
        raise AssertionError("fresh SCF must not run with global module hooks")

    register = getattr(torch_module, register_name)
    hook = register(lambda *_args, **_kwargs: None)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(scf.hf.RHF, "kernel", unexpected_kernel, raising=False)
            with pytest.raises(
                RHFDeePHFScannerError,
                match="cannot contain module execution hooks",
            ):
                scanner(REFERENCE_COORDINATES)
    finally:
        hook.remove()

    assert scf_calls == []
    _assert_no_current_result(scanner)
    assert scanner._root_anchor is anchor_before


def test_root_continuity_failure_does_not_publish_or_advance_and_recovers(
    scanner_case,
    monkeypatch,
):
    scanner = scanner_case.scanner
    scanner(REFERENCE_COORDINATES)
    anchor_before = scanner._root_anchor
    factory = scanner._reference_factory
    original_occupied_overlap = factory._occupied_overlap

    def reject_root(*_args, **_kwargs):
        raise RHFScannerReferenceError(
            "fresh scanner RHF occupied subspace is discontinuous"
        )

    displaced = REFERENCE_COORDINATES.copy()
    displaced[1, 1] += 0.03
    monkeypatch.setattr(factory, "_occupied_overlap", reject_root)
    with pytest.raises(RHFScannerReferenceError, match="discontinuous"):
        scanner(displaced)

    _assert_no_current_result(scanner)
    assert scanner._root_anchor is anchor_before

    monkeypatch.setattr(factory, "_occupied_overlap", original_occupied_overlap)
    energy, gradient = scanner(displaced)
    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    assert scanner.converged
    assert scanner._root_anchor is not anchor_before


def test_unconverged_fresh_scf_clears_state_preserves_anchor_and_recovers(
    scanner_case,
    monkeypatch,
):
    scanner = scanner_case.scanner
    scanner(REFERENCE_COORDINATES)
    anchor_before = scanner._root_anchor

    def unconverged_kernel(reference, dm0=None, **_kwargs):
        assert dm0 is None
        reference.converged = False
        return 0.0

    displaced = REFERENCE_COORDINATES.copy()
    displaced[1, 0] -= 0.025
    with monkeypatch.context() as patch:
        patch.setattr(
            scf.hf.RHF,
            "kernel",
            unconverged_kernel,
            raising=False,
        )
        with pytest.raises(
            DeePHFCapabilityError,
            match="fresh scanner RHF reference did not converge",
        ):
            scanner(displaced)

    _assert_no_current_result(scanner)
    assert scanner._root_anchor is anchor_before
    energy, gradient = scanner(displaced)
    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    assert scanner.converged
    assert scanner._root_anchor is not anchor_before


def test_scanner_recomputes_after_a_legal_between_call_model_change(scanner_case):
    scanner = scanner_case.scanner
    first_energy, first_gradient = scanner(REFERENCE_COORDINATES)
    first_fingerprint = scanner.model_state_fingerprint
    first_method = scanner.method

    with torch.no_grad():
        scanner_case.model.linear.weight.add_(0.008)

    second_energy, second_gradient = scanner(REFERENCE_COORDINATES)
    fresh_energy, fresh_gradient = _fresh_deephf_result(
        REFERENCE_COORDINATES,
        scanner_case.model,
    )

    assert scanner.model_state_fingerprint != first_fingerprint
    assert scanner.method is not first_method
    assert abs(second_energy - first_energy) > 1.0e-6
    assert np.max(np.abs(second_gradient - first_gradient)) > 1.0e-7
    np.testing.assert_allclose(second_energy, fresh_energy, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        second_gradient,
        fresh_gradient,
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_scanner_detects_a_model_change_during_evaluation_and_recovers(
    scanner_case,
    monkeypatch,
):
    scanner = scanner_case.scanner
    anchor_before = scanner._root_anchor
    original_forward = CorrNet.forward

    def mutate_model(model, inputs):
        output = original_forward(model, inputs)
        with torch.no_grad():
            model.linear.bias.add_(1.0e-5)
        return output

    with monkeypatch.context() as patch:
        patch.setattr(CorrNet, "forward", mutate_model)
        with pytest.raises(
            RHFDeePHFScannerError,
            match="forward implementation was replaced",
        ):
            scanner(REFERENCE_COORDINATES)

    _assert_no_current_result(scanner)
    assert scanner._root_anchor is anchor_before
    energy, gradient = scanner(REFERENCE_COORDINATES)
    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()


def test_scanner_rejects_a_training_submodule_before_fresh_scf_and_recovers(
    scanner_case,
    monkeypatch,
):
    scanner = scanner_case.scanner
    anchor_before = scanner._root_anchor
    scf_calls = []

    def unexpected_kernel(*args, **kwargs):
        scf_calls.append((args, kwargs))
        raise AssertionError("fresh SCF must not run with a training submodule")

    scanner_case.model.linear.train(True)
    with monkeypatch.context() as patch:
        patch.setattr(scf.hf.RHF, "kernel", unexpected_kernel, raising=False)
        with pytest.raises(
            RHFDeePHFScannerError,
            match="must remain in evaluation mode.*linear",
        ):
            scanner(REFERENCE_COORDINATES)

    assert scf_calls == []
    assert not scanner_case.model.training
    assert scanner_case.model.linear.training
    _assert_no_current_result(scanner)
    assert scanner._root_anchor is anchor_before
    scanner_case.model.eval()
    energy, gradient = scanner(REFERENCE_COORDINATES)
    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    assert scanner.converged


def test_scanner_rejects_paired_mode_restoring_hooks_without_advancing_root(
    scanner_case,
    monkeypatch,
):
    scanner = scanner_case.scanner
    scanner(REFERENCE_COORDINATES)
    anchor_before = scanner._root_anchor
    hook_calls = []
    scf_calls = []

    def enable_training(module, _inputs):
        hook_calls.append("pre")
        module.train(True)

    def restore_evaluation(module, _inputs, _output):
        hook_calls.append("post")
        module.eval()

    def unexpected_kernel(*args, **kwargs):
        scf_calls.append((args, kwargs))
        raise AssertionError("fresh SCF must not run with paired module hooks")

    pre_hook = scanner_case.model.register_forward_pre_hook(enable_training)
    post_hook = scanner_case.model.register_forward_hook(restore_evaluation)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(scf.hf.RHF, "kernel", unexpected_kernel, raising=False)
            with pytest.raises(
                RHFDeePHFScannerError,
                match="cannot contain module execution hooks",
            ):
                scanner(REFERENCE_COORDINATES)
    finally:
        pre_hook.remove()
        post_hook.remove()

    assert hook_calls == []
    assert scf_calls == []
    assert not scanner_case.model.training
    _assert_no_current_result(scanner)
    assert scanner._root_anchor is anchor_before

    displaced = REFERENCE_COORDINATES.copy()
    displaced[1, 0] += 0.02
    energy, gradient = scanner(displaced)
    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    assert scanner.converged
    assert scanner._root_anchor is not anchor_before


def test_scanner_failure_clears_results_does_not_advance_root_and_recovers(
    scanner_case,
):
    scanner = scanner_case.scanner
    scanner(REFERENCE_COORDINATES)
    anchor_before = scanner._root_anchor
    saved_weight = scanner_case.model.linear.weight.detach().clone()
    displaced = REFERENCE_COORDINATES.copy()
    displaced[1, 2] += 0.03

    with torch.no_grad():
        scanner_case.model.linear.weight.fill_(float("nan"))
    with pytest.raises(DeePHFCapabilityError, match="must be finite"):
        scanner(displaced)

    _assert_no_current_result(scanner)
    assert scanner.base is scanner_case.driver
    assert scanner._root_anchor is anchor_before

    with torch.no_grad():
        scanner_case.model.linear.weight.copy_(saved_weight)
    energy, gradient = scanner(displaced)
    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    assert scanner.converged
    assert scanner._root_anchor is not anchor_before


def test_scanner_always_starts_a_new_native_rhf_without_a_previous_density(
    scanner_case,
    monkeypatch,
):
    original_kernel = scf.hf.SCF.kernel
    calls = []

    def tracked_kernel(reference, dm0=None, **kwargs):
        calls.append((reference, dm0, reference.mo_coeff))
        return original_kernel(reference, dm0=dm0, **kwargs)

    monkeypatch.setattr(scf.hf.RHF, "kernel", tracked_kernel, raising=False)
    first = REFERENCE_COORDINATES.copy()
    second = REFERENCE_COORDINATES.copy()
    second[1, 0] += 0.02

    scanner_case.scanner(first)
    scanner_case.scanner(second)

    assert len(calls) == 2
    assert calls[0][0] is not calls[1][0]
    assert all(dm0 is None for _, dm0, _ in calls)
    assert all(coefficients is None for _, _, coefficients in calls)


def test_scanner_rejects_invalid_or_static_incompatible_inputs_without_stale_state(
    scanner_case,
):
    scanner = scanner_case.scanner
    scanner(REFERENCE_COORDINATES)
    invalid_coordinates = REFERENCE_COORDINATES.copy()
    invalid_coordinates[0, 0] = np.nan
    invalid_inputs = (
        np.zeros((1, 3), dtype=np.float64),
        invalid_coordinates,
        REFERENCE_COORDINATES.astype(np.complex128),
        object(),
        _molecule(REFERENCE_COORDINATES, basis="6-31g"),
    )

    for invalid_input in invalid_inputs:
        with pytest.raises((TypeError, ValueError, RHFScannerReferenceError)):
            scanner(invalid_input)
        _assert_no_current_result(scanner)

    energy, gradient = scanner(REFERENCE_COORDINATES)
    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    assert scanner.converged


def test_scanner_rejects_an_instance_coordinate_hook_before_fresh_scf(
    scanner_case,
    monkeypatch,
):
    scanner = scanner_case.scanner
    true_coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        dtype=np.float64,
    )
    forged_coordinates = true_coordinates.copy()
    forged_coordinates[1, 2] = 1.6
    external_molecule = _molecule(true_coordinates)
    anchor_before = scanner._root_anchor
    hook_calls = []
    scf_calls = []

    def forged_atom_coords(*, unit="Bohr"):
        hook_calls.append(unit)
        return forged_coordinates.copy()

    def unexpected_kernel(*args, **kwargs):
        scf_calls.append((args, kwargs))
        raise AssertionError("fresh SCF must not run after an instance hook")

    external_molecule.atom_coords = forged_atom_coords
    with monkeypatch.context() as patch:
        patch.setattr(scf.hf.RHF, "kernel", unexpected_kernel, raising=False)
        with pytest.raises(
            RHFScannerReferenceError,
            match="callable instance hooks.*atom_coords",
        ):
            scanner(external_molecule)

    assert hook_calls == []
    assert scf_calls == []
    _assert_no_current_result(scanner)
    assert scanner._root_anchor is anchor_before

    del external_molecule.atom_coords
    input_fingerprint = molecule_science_fingerprint(external_molecule)
    input_coordinates = external_molecule.atom_coords(unit="Bohr").copy()
    energy, gradient = scanner(external_molecule)

    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    assert scanner.converged
    assert scanner._root_anchor is not anchor_before
    assert molecule_science_fingerprint(external_molecule) == input_fingerprint
    np.testing.assert_array_equal(
        external_molecule.atom_coords(unit="Bohr"),
        input_coordinates,
    )
