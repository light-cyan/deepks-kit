import copy

import numpy as np
import pytest
from pyscf import gto, scf

from deepks.deephf import (
    DeePHF,
    DeePHFCapabilityError,
    RHFResponseError,
    generate_rhf_force_frame,
    write_rhf_force_dataset,
)
from deepks.data.force_schema import load_force_dataset
from deepks.descriptor import DescriptorDifferentiabilityError

FINITE_DIFFERENCE_ATOM = 0
FINITE_DIFFERENCE_COORDINATE = 0
ORACLE_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]


def _generate_frame(case):
    return generate_rhf_force_frame(
        case.reference,
        projector_basis=ORACLE_PROJECTOR_BASIS,
        e_target=case.target_energy,
        f_target=case.target_force,
    )


def test_generated_frame_is_the_direct_relaxed_oracle(
    force_generation_case,
    generated_force_frame,
):
    case = force_generation_case
    frame = generated_force_frame
    arrays = frame.arrays
    teacher_gradient = case.teacher_gradient

    assert set(arrays) == {
        "atom",
        "descriptor",
        "e_base",
        "f_base",
        "e_target",
        "f_target",
        "e_corr_target",
        "f_corr_target",
        "dq_dR_relaxed",
    }
    assert all(value.dtype == np.dtype(np.float64) for value in arrays.values())
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert all(not value.flags.writeable for value in arrays.values())
    assert arrays["atom"].shape == (3, 4)
    assert arrays["descriptor"].shape == (3, 4)
    assert arrays["dq_dR_relaxed"].shape == (3, 3, 3, 4)

    np.testing.assert_allclose(
        arrays["descriptor"],
        case.teacher_method.descriptor(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        arrays["dq_dR_relaxed"],
        teacher_gradient.dq_dR_relaxed,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        arrays["dq_dR_relaxed"],
        teacher_gradient.dq_dR_explicit + teacher_gradient.dq_dR_response,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.max(
        np.abs(arrays["dq_dR_relaxed"] - teacher_gradient.dq_dR_explicit)
    ) > 0.1

    np.testing.assert_allclose(arrays["e_target"], case.target_energy)
    np.testing.assert_allclose(arrays["f_target"], case.target_force)
    np.testing.assert_allclose(
        arrays["e_corr_target"],
        case.teacher_method.e_corr,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        arrays["f_corr_target"],
        -teacher_gradient.correction_gradient,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    provenance = frame.provenance
    assert provenance["reference"]["method"] == "RHF"
    assert provenance["reference"]["converged"] is True
    assert provenance["response"]["backend"] == "pyscf-2.14-rhf-direct"
    assert provenance["response"]["converged"] is True
    diagnostics = provenance["response"]["diagnostics"]
    assert diagnostics["maximum_residual"] <= diagnostics["residual_tolerance"]
    assert provenance["descriptor"]["differentiability"][
        "structural_zero_blocks"
    ] == []


def test_relaxed_jacobian_and_target_force_have_finite_difference_sign(
    force_generation_case,
    generated_force_frame,
):
    case = force_generation_case
    frame = generated_force_frame
    atom_index = FINITE_DIFFERENCE_ATOM
    coordinate_index = FINITE_DIFFERENCE_COORDINATE

    np.testing.assert_allclose(
        frame.arrays["dq_dR_relaxed"][atom_index, coordinate_index],
        case.descriptor_finite_difference,
        rtol=2.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        frame.arrays["f_corr_target"][atom_index, coordinate_index],
        -case.correction_energy_finite_difference,
        rtol=3.0e-6,
        atol=1.0e-7,
    )


def test_structural_zero_descriptor_blocks_are_not_persisted(
):
    molecule = gto.M(
        atom="H 0.1 0.2 0.3; H -0.2 0.4 1.5",
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        cart=False,
        verbose=0,
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged

    with pytest.raises(
        DescriptorDifferentiabilityError,
        match="structural zero blocks",
    ):
        generate_rhf_force_frame(
            reference,
            projector_basis=[[1, [0.8, 1.0]]],
            e_target=np.float64(reference.e_tot),
            f_target=np.zeros((2, 3), dtype=np.float64),
        )


def test_multiframe_failure_does_not_create_a_partial_dataset(
    tmp_path,
    force_generation_case,
):
    case = force_generation_case
    invalid_reference = copy.copy(case.reference)
    invalid_reference.converged = False
    output = tmp_path / "force-data"

    with pytest.raises(DeePHFCapabilityError, match="must be converged"):
        write_rhf_force_dataset(
            output,
            [case.reference, invalid_reference],
            projector_basis=ORACLE_PROJECTOR_BASIS,
            e_target=np.array(
                [case.target_energy, invalid_reference.e_tot],
                dtype=np.float64,
            ),
            f_target=np.stack(
                [case.target_force, case.target_force],
                axis=0,
            ),
        )

    assert not output.exists()


def test_response_failure_does_not_fall_back_or_create_dataset(
    tmp_path,
    force_generation_case,
):
    case = force_generation_case
    output = tmp_path / "failed-response-data"

    with pytest.raises(RHFResponseError, match="residual"):
        write_rhf_force_dataset(
            output,
            case.reference,
            projector_basis=ORACLE_PROJECTOR_BASIS,
            e_target=case.target_energy,
            f_target=case.target_force,
            response_options={
                "residual_tolerance": 1.0e-30,
                "max_refinement_cycles": 0,
            },
        )

    assert not output.exists()


def test_multiframe_direct_dataset_round_trip(
    tmp_path,
    force_generation_case,
    generated_force_frame,
):
    case = force_generation_case
    forward_teacher = DeePHF(
        case.forward_reference,
        case.teacher_method.model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    forward_energy = np.float64(forward_teacher.kernel())
    forward_gradient = forward_teacher.nuc_grad_method().run()
    forward_force = np.asarray(-forward_gradient.de_full, dtype=np.float64)
    output = tmp_path / "multi-frame-force-data"

    contract = write_rhf_force_dataset(
        output,
        [case.reference, case.forward_reference],
        projector_basis=ORACLE_PROJECTOR_BASIS,
        e_target=np.array([case.target_energy, forward_energy], dtype=np.float64),
        f_target=np.stack([case.target_force, forward_force], axis=0),
    )
    loaded_contract, arrays = load_force_dataset(output)

    assert contract.manifest_fingerprint == loaded_contract.manifest_fingerprint
    assert contract.dimensions == {
        "n_frame": 2,
        "n_raw_atom": 3,
        "n_descriptor_atom": 3,
        "n_feature": 4,
    }
    assert not (output / "dq_dR_explicit.npy").exists()
    assert set(path.name for path in output.iterdir()) == {
        "atom.npy",
        "descriptor.npy",
        "e_base.npy",
        "f_base.npy",
        "e_target.npy",
        "f_target.npy",
        "e_corr_target.npy",
        "f_corr_target.npy",
        "dq_dR_relaxed.npy",
        "force_data.json",
    }
    np.testing.assert_allclose(
        arrays["dq_dR_relaxed"][0],
        generated_force_frame.arrays["dq_dR_relaxed"],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        arrays["dq_dR_relaxed"][1],
        forward_gradient.dq_dR_relaxed,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        arrays["e_target"],
        [[case.target_energy], [forward_energy]],
    )
    np.testing.assert_allclose(
        arrays["f_target"],
        np.stack([case.target_force, forward_force], axis=0),
    )
    manifest = loaded_contract.manifest
    assert manifest["fields"]["dq_dR_relaxed"]["semantics"] == (
        "complete_relaxed_reference_response"
    )
    assert manifest["generation"]["producer"] == (
        "deepks.deephf.force_data.rhf_direct"
    )
    for frame in manifest["frames"]:
        diagnostics = frame["response_diagnostics"]
        assert diagnostics["maximum_residual"] <= diagnostics["residual_tolerance"]
