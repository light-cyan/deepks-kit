import numpy as np
import pytest
import torch
from pyscf import gto

from deepks.deephf import DeePHF, build_reference, make_deephf
from deepks.deephf.capabilities import DeePHFCapabilityError
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]]]


def _model():
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
        model.linear.weight.fill_(0.23)
        model.linear.bias.fill_(0.17)
    return model.eval()


def _rhf_method():
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


def _uhf_method():
    molecule = gto.M(
        atom="Li 0 0 0",
        basis="sto-3g",
        unit="Bohr",
        spin=1,
        verbose=0,
    )
    return make_deephf(
        build_reference(molecule, "uhf"),
        _model(),
        projector_basis=PROJECTOR_BASIS,
    )


def _make_writable_and_change(value):
    value.setflags(write=True)
    value.fill(91.0)


def test_public_rhf_arrays_do_not_alias_calculation_caches_or_change_direct_gradient():
    method = _rhf_method()
    with method.calculation():
        energy = method.kernel()
        context = method._context()
        public_density = method.ao_density()
        public_projected = method.projected_density(flatten=True)
        public_descriptor = method.descriptor()
        public_sensitivity = method.correction_sensitivity()
        public_dq_dp = method.dq_dP()
        cached_arrays = (
            context.density,
            torch.cat(
                [block.flatten(-2) for block in context.workspace.projected_blocks],
                dim=-1,
            ).detach().cpu().numpy(),
            context.descriptor_values.detach().cpu().numpy(),
            context.sensitivity,
            context.workspace.cached_dq_dP,
        )
        public_arrays = (
            public_density,
            public_projected,
            public_descriptor,
            public_sensitivity,
            public_dq_dp,
        )
        snapshots = tuple(np.array(value, copy=True) for value in cached_arrays)
        assert all(
            not np.shares_memory(public, cached)
            for public, cached in zip(public_arrays, cached_arrays, strict=True)
        )
        for value in public_arrays:
            _make_writable_and_change(value)
        for cached, snapshot in zip(cached_arrays, snapshots, strict=True):
            np.testing.assert_array_equal(cached, snapshot)
        gradient = method.nuc_grad_method(
            backend="direct",
            retain_details=False,
        ).kernel()
        assert method.correction_energy() == energy - method.e_base
        np.testing.assert_array_equal(method.descriptor(), snapshots[2])
    assert np.isfinite(gradient).all()
    assert method.operation_counts["descriptor_evaluations"] == 1
    assert method.operation_counts["model_forwards"] == 1
    assert method.operation_counts["direct_response_solves"] == 1


def test_public_uhf_sensitivity_cannot_trigger_zvector_zero_fast_path():
    method = _uhf_method()
    with method.calculation():
        method.kernel()
        public_spin_density = method.spin_ao_density()
        public_sensitivity = method.correction_sensitivity()
        cached_spin_density = method._context().spin_density
        cached_sensitivity = method._context().sensitivity
        assert not np.shares_memory(public_spin_density, cached_spin_density)
        assert not np.shares_memory(public_sensitivity, cached_sensitivity)
        _make_writable_and_change(public_spin_density)
        public_sensitivity.setflags(write=True)
        public_sensitivity.fill(0.0)
        assert np.any(cached_sensitivity)
        gradient = method.nuc_grad_method(
            backend="zvector",
            retain_details=False,
        ).kernel()
    assert np.isfinite(gradient).all()
    assert method.operation_counts["adjoint_solves"] == 1
    assert method.operation_counts["derivative_overlap_integral_evaluations"] == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "model_parameter",
        "model_mode",
        "model_dtype",
        "reference_orbital",
        "reference_occupation",
        "reference_energy",
        "molecule_state",
        "descriptor_state",
        "device",
    ),
)
def test_mid_transaction_mutation_fails_before_cached_reuse(mutation):
    method = _rhf_method()
    with pytest.raises(DeePHFCapabilityError, match="scientific state changed"):
        with method.calculation():
            method.kernel()
            if mutation == "model_parameter":
                with torch.no_grad():
                    method.model.linear.weight.add_(0.01)
            elif mutation == "model_mode":
                method.model.train()
            elif mutation == "model_dtype":
                method.model.float()
            elif mutation == "reference_orbital":
                method.reference.mo_coeff[0, 0] += 0.01
            elif mutation == "reference_occupation":
                method.reference.mo_occ[0] -= 0.01
            elif mutation == "reference_energy":
                method.reference.e_tot += 0.01
            elif mutation == "molecule_state":
                method.mol._env[-1] += 0.01
            elif mutation == "descriptor_state":
                with torch.no_grad():
                    method._descriptor.overlap_shells[0].add_(0.01)
            else:
                method.device = "meta"
            method.descriptor()
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)
    assert method.operation_counts["state_version_validations"] <= 1
    assert method.operation_counts["cache_invalidations"] == 1


@pytest.mark.parametrize("backend", ["direct", "zvector"])
def test_exit_failure_atomically_clears_method_and_gradient_results(backend):
    method = _rhf_method()
    driver = method.nuc_grad_method(backend=backend, retain_details=True)
    with pytest.raises(DeePHFCapabilityError, match="scientific state changed"):
        with method.calculation():
            method.kernel()
            driver.kernel()
            assert method.e_tot is not None
            assert driver.de is not None
            with torch.no_grad():
                method.model.linear.bias.add_(0.01)
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)
    assert driver.de is None
    assert driver.descriptor_diagnostics is None
    assert driver.response_diagnostics is None
    assert not hasattr(driver, "de_full")
    assert not hasattr(driver, "response_result")
    assert not hasattr(driver, "adjoint_result")


def test_outer_transaction_entry_failure_clears_previous_method_results():
    method = _rhf_method()
    assert np.isfinite(method.kernel())
    assert method.e_tot is not None
    method.reference.e_tot += 0.01

    with pytest.raises(DeePHFCapabilityError, match="scientific state changed"):
        with method.calculation():
            pass
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)
