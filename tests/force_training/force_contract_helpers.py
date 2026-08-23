"""Helpers for synthetic strict force-contract tests."""

from __future__ import annotations

import numpy as np
import torch

from deepks.data.force_schema import _write_force_dataset
from deepks.model.reader import Reader


def _as_float64_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def write_force_contract_sample(
    directory,
    *,
    energy,
    descriptor,
    force,
    jacobian,
    projector_basis,
    shell_sizes,
):
    """Persist synthetic values through the same strict contract used at runtime."""
    energy = _as_float64_numpy(energy)
    descriptor = _as_float64_numpy(descriptor)
    force = _as_float64_numpy(force)
    jacobian = _as_float64_numpy(jacobian)
    frame_count, descriptor_atom_count, feature_count = descriptor.shape
    raw_atom_count = force.shape[1]
    atom = np.zeros((frame_count, raw_atom_count, 4), dtype=np.float64)
    atom[:, :, 0] = 1.0
    atom[:, :, 1:] = np.arange(
        frame_count * raw_atom_count * 3,
        dtype=np.float64,
    ).reshape(frame_count, raw_atom_count, 3) / 100.0
    zeros_energy = np.zeros((frame_count, 1), dtype=np.float64)
    zeros_force = np.zeros((frame_count, raw_atom_count, 3), dtype=np.float64)
    arrays = {
        "atom": atom,
        "descriptor": descriptor,
        "e_base": zeros_energy,
        "f_base": zeros_force,
        "e_target": energy,
        "f_target": force,
        "e_corr_target": energy,
        "f_corr_target": force,
        "dq_dR_relaxed": jacobian,
    }
    response_controls = {
        "cphf_tolerance": 1.0e-12,
        "residual_tolerance": 1.0e-9,
        "invariant_tolerance": 1.0e-9,
        "orbital_gap_tolerance": 1.0e-8,
        "max_cycle": 50,
        "max_refinement_cycles": 3,
        "level_shift": 0.0,
        "operator_stability_tolerance": 1.0e-6,
        "operator_condition_tolerance": 1.0e8,
        "operator_symmetry_tolerance": 1.0e-10,
        "operator_dimension_limit": 512,
    }
    frames = []
    for frame_index in range(frame_count):
        maximum_residual = 1.0e-11 + frame_index * 1.0e-12
        frames.append(
            {
                "reference_state_fingerprint": f"{frame_index + 1:064x}",
                "reference_converged": True,
                "response_converged": True,
                "response_integrity_fingerprint": f"{frame_index + 101:064x}",
                "response_diagnostics": {
                    "minimum_orbital_gap": 0.5,
                    "pyscf_version": "2.14.0",
                    "cphf_tolerance": response_controls["cphf_tolerance"],
                    "maximum_residual": maximum_residual,
                    "residual_rms": maximum_residual / 2.0,
                    "residual_tolerance": response_controls["residual_tolerance"],
                    "invariant_tolerance": response_controls["invariant_tolerance"],
                    "orbital_gap_tolerance": response_controls["orbital_gap_tolerance"],
                    "max_cycle": response_controls["max_cycle"],
                    "max_refinement_cycles": response_controls["max_refinement_cycles"],
                    "level_shift": response_controls["level_shift"],
                    "response_dimension": 1,
                    "operator_stability_tolerance": response_controls[
                        "operator_stability_tolerance"
                    ],
                    "operator_condition_tolerance": response_controls[
                        "operator_condition_tolerance"
                    ],
                    "operator_symmetry_tolerance": response_controls[
                        "operator_symmetry_tolerance"
                    ],
                    "operator_dimension_limit": response_controls[
                        "operator_dimension_limit"
                    ],
                    "operator_diagnostics_are_estimates": True,
                    "operator_minimum_eigenvalue": 0.25,
                    "operator_maximum_eigenvalue": 2.0,
                    "operator_condition_number": 8.0,
                    "operator_symmetry_residual": 1.0e-15,
                    "metric_residual": 1.0e-14,
                    "idempotency_residual": 1.0e-14,
                    "particle_number_residual": 1.0e-14,
                    "refinement_cycles": 0,
                    "residual_history": [maximum_residual],
                },
                "descriptor_diagnostics": {
                    "minimum_scaled_gap": 10.0 + frame_index,
                    "structural_zero_blocks": [],
                },
                "geometry_bohr": atom[frame_index, :, 1:].tolist(),
            }
        )
    provenance = {
        "atom_mapping": {
            "descriptor_to_raw": list(range(descriptor_atom_count)),
            "raw_to_descriptor": list(range(raw_atom_count)),
            "nuclear_charges": [1] * raw_atom_count,
            "ghost_policy": "rejected",
        },
        "descriptor": {
            "definition": "ordered_projected_density_eigenvalues",
            "spin_semantics": "spin_summed",
            "shell_sizes": list(shell_sizes),
            "projector_basis": projector_basis,
            "differentiability_controls": {
                "gap_atol": 1.0e-9,
                "gap_rtol": 1.0e-7,
                "zero_atol": 1.0e-9,
                "sensitivity_atol": 1.0e-8,
                "structural_zero_blocks": "rejected",
                "validator": "deepks.descriptor.validate_differentiability",
            },
        },
        "reference": {
            "family": "RHF",
            "python_class": "pyscf.scf.hf.RHF",
            "basis_content": {"H": [[0, [1.24, 1.0]]]},
            "ecp": None,
            "charge": 0,
            "spin": 0,
            "occupations": [2.0, 0.0],
            "scf_controls": {
                "conv_tol": 1.0e-13,
                "conv_tol_grad": None,
                "conv_tol_cpscf": 1.0e-9,
                "max_cycle": 100,
                "level_shift": 0.0,
                "diis_space": 8,
                "direct_scf": True,
                "conv_check": True,
            },
        },
        "response": {
            "backend": "rhf_direct",
            "adapter": "deepks.deephf.pyscf_rhf.RHFResponseAdapter",
            "controls": response_controls,
        },
        "frames": frames,
        "generation": {
            "deepks_version": "0.1.dev-test",
            "deepks_commit": None,
            "pyscf_version": "2.14.0",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": "3.11.test",
            "producer": "deepks.deephf.force_data.rhf_direct",
            "producer_version": 1,
        },
    }
    contract = _write_force_dataset(
        directory,
        arrays=arrays,
        provenance=provenance,
    )
    reader = Reader(
        directory,
        batch_size=frame_count,
        force_mode="deephf_relaxed",
    )
    return contract, reader.sample_all()
