import numpy as np
import torch
from pyscf import gto

from deepks.deephf import DeePHF, build_reference
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]]]


def make_method(*, constant=False, model_none=False):
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    reference = build_reference(molecule, "rhf")
    if model_none:
        model = None
    else:
        model = CorrNet(
            input_dim=1,
            hidden_sizes=(2,),
            proj_basis=PROJECTOR_BASIS,
        ).double()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.zero_()
            model.linear.bias.fill_(0.17)
            if not constant:
                model.linear.weight.fill_(0.23)
        model.eval()
    return DeePHF(reference, model, projector_basis=PROJECTOR_BASIS)


def run_public_sequence(method, backend="direct"):
    with method.calculation():
        method.kernel()
        driver = method.nuc_grad_method(backend=backend, retain_details=False)
        gradient = driver.kernel()
        descriptor = method.descriptor()
    assert np.isfinite(gradient).all()
    assert np.isfinite(descriptor).all()
    counts = method.operation_counts
    # Construction performs one additional full binding fingerprint outside this scope.
    assert counts["science_state_fingerprints"] <= 2
    assert counts["state_version_validations"] <= 2
    assert counts["cache_state_fingerprints"] <= 2
    assert counts["cache_hits"] > 0
    assert counts.get("cache_invalidations", 0) == 0
    return counts


def test_fresh_public_context_does_not_fingerprint_an_empty_cache():
    method = make_method()
    with method.calculation():
        method.kernel()
        assert method._active_operation_counts.get("cache_state_fingerprints", 0) == 0
        assert method._active_operation_counts.get("state_version_validations", 0) == 0


def test_controlled_workflow_has_no_intermediate_cache_state_scans():
    method = make_method()
    with method._controlled_calculation():
        method.kernel()
        gradient = method.nuc_grad_method(
            backend="direct",
            retain_details=False,
        ).kernel()
        descriptor = method.descriptor()
    assert np.isfinite(gradient).all()
    assert np.isfinite(descriptor).all()
    counts = method.operation_counts
    assert counts["science_state_fingerprints"] <= 2
    assert counts.get("state_version_validations", 0) == 0
    assert counts.get("cache_state_fingerprints", 0) == 0
    assert counts["descriptor_evaluations"] == 1
    assert counts["model_forwards"] == 1


def test_nonzero_public_sequence_reuses_energy_descriptor_and_derivatives():
    counts = run_public_sequence(make_method())
    assert counts["ao_density_constructions"] == 1
    assert counts["descriptor_evaluations"] == 1
    assert counts["model_forwards"] == 1
    assert counts["projected_density_constructions"] == 1
    assert counts["shell_eigenvalue_jacobian_constructions"] == 1
    assert counts["derivative_overlap_integral_evaluations"] == 1
    assert counts["direct_response_solves"] == 1
    assert counts["response_operator_actions"] == 2
    assert counts.get("preconditioner_actions", 0) == 0
    assert counts.get("adjoint_solves", 0) == 0
    assert counts.get("full_density_partition_materializations", 0) == 0


def test_zero_sensitivity_compact_backends_skip_derivatives_and_response():
    for backend in ("direct", "zvector"):
        counts = run_public_sequence(make_method(constant=True), backend)
        assert counts["descriptor_evaluations"] == 1
        assert counts["model_forwards"] == 1
        assert counts.get("shell_eigenvalue_jacobian_constructions", 0) == 0
        assert counts.get("derivative_overlap_integral_evaluations", 0) == 0
        assert counts.get("direct_response_solves", 0) == 0
        assert counts.get("adjoint_solves", 0) == 0
        assert counts.get("response_operator_actions", 0) == 0
        assert counts.get("preconditioner_actions", 0) == 0
        assert counts.get("full_density_partition_materializations", 0) == 0


def test_nonzero_zvector_sequence_has_one_bounded_adjoint_solve():
    counts = run_public_sequence(make_method(), backend="zvector")
    assert counts["descriptor_evaluations"] == 1
    assert counts["model_forwards"] == 1
    assert counts["adjoint_solves"] == 1
    assert counts["response_operator_actions"] == 1
    assert counts["preconditioner_actions"] == 1
    assert counts.get("direct_response_solves", 0) == 0
    assert counts.get("full_density_partition_materializations", 0) == 0


def test_detailed_force_data_path_keeps_model_independent_response_materialization():
    method = make_method(model_none=True)
    with method.calculation():
        method.kernel()
        driver = method.nuc_grad_method(backend="direct", retain_details=True)
        gradient = driver.kernel()
        descriptor = method.descriptor()
    assert np.isfinite(gradient).all()
    assert np.isfinite(descriptor).all()
    counts = method.operation_counts
    assert counts["descriptor_evaluations"] == 1
    assert counts.get("model_forwards", 0) == 0
    assert counts["projected_density_constructions"] == 1
    assert counts["shell_eigenvalue_jacobian_constructions"] == 1
    assert counts["derivative_overlap_integral_evaluations"] == 1
    assert counts["direct_response_solves"] == 1
    assert counts["full_density_partition_materializations"] == 1
    assert counts["response_operator_actions"] == 2
