import numpy as np

from scripts.analyze_energy_stability import (
    plot_energy,
    plot_timing,
    stable_duration,
    validate_common_protocol,
    validate_energy_series,
    write_report,
)


def test_stable_duration_returns_full_duration_without_threshold_crossing():
    times = np.asarray([0.0, 0.25, 0.5])
    drift = np.asarray([0.0, 0.2, -0.9])

    assert stable_duration(times, drift) == 0.5


def test_stable_duration_stops_at_frame_before_first_crossing():
    times = np.asarray([0.0, 0.25, 0.5, 0.75])
    drift = np.asarray([0.0, 0.8, 1.1, 0.2])

    assert stable_duration(times, drift) == 0.25


def test_stable_duration_is_zero_when_initial_frame_is_outside_band():
    times = np.asarray([0.0, 0.25])
    drift = np.asarray([1.1, 0.0])

    assert stable_duration(times, drift) == 0.0


def _energy_series():
    potential = np.asarray([-2.0, -1.9, -1.8])
    kinetic = np.asarray([0.1, 0.2, 0.3])
    total = potential + kinetic
    drift = total - total[0]
    return {
        "step": np.asarray([0.0, 1.0, 2.0]),
        "time_fs": np.asarray([0.0, 0.25, 0.5]),
        "potential_energy_eV": potential,
        "kinetic_energy_eV": kinetic,
        "total_energy_eV": total,
        "delta_total_energy_eV": drift,
        "delta_total_energy_eV_per_atom": drift / 2,
        "temperature_K": np.asarray([50.0, 60.0, 70.0]),
    }


def test_energy_series_requires_complete_consistent_total_energy():
    energy = _energy_series()

    validate_energy_series(energy, atoms=2, steps=2, timestep_fs=0.25)

    energy["total_energy_eV"][1] += 0.1
    with np.testing.assert_raises_regex(ValueError, "potential plus kinetic"):
        validate_energy_series(energy, atoms=2, steps=2, timestep_fs=0.25)


def test_common_protocol_rejects_mixed_xc():
    first = _report_result()
    second = {**first, "system": "gram_02", "xc": "PBE"}

    with np.testing.assert_raises_regex(ValueError, "different protocol: xc"):
        validate_common_protocol([first, second])


def _report_result():
    return {
        "system": "gram_01",
        "configuration": "gram_01_rxn000026_p000026_0",
        "formula": "C2H4",
        "atoms": 6,
        "charge": 0,
        "multiplicity": 1,
        "reference_family": "RKS",
        "xc": "B3LYP5",
        "basis": "def2-tzvp",
        "grid_mode": "default",
        "grid_level": 3,
        "small_rho_cutoff": 0.0,
        "model": "/runtime/b3lyp_gram_t1x.pth",
        "model_name": "b3lyp_gram_t1x.pth",
        "force_mode": "analytic",
        "finite_difference_step_bohr": None,
        "target_temperature_K": 100.0,
        "initial_temperature_K": 66.7,
        "timestep_fs": 0.25,
        "steps": 400,
        "slurm_job_id": "1023_0",
        "simulated_duration_fs": 100.0,
        "stable_duration_at_1meV_per_atom_fs": 100.0,
        "stable_through_full_run": True,
        "maximum_absolute_drift_meV_per_atom": 0.012345,
        "rms_drift_meV_per_atom": 0.006789,
        "final_drift_meV_per_atom": -0.003210,
        "linear_drift_meV_per_atom_per_fs": 1.2e-5,
        "md_wall_seconds_per_simulated_fs": 250.0,
        "total_wall_seconds_per_simulated_fs": 260.0,
    }


def test_report_marks_full_run_stability_and_links_plots(tmp_path):
    result = _report_result()
    output = tmp_path / "report.md"

    write_report([result], output)

    report = output.read_text(encoding="utf-8")
    assert (
        "| gram_01 | C2H4 | 6 | 0 | 1 | GPU4PySCF RKS | B3LYP5 | "
        "def2-tzvp | default/level-3/rho-0 | b3lyp_gram_t1x.pth | analytic | "
        "100.0 | 66.7 | 0.250 | 400 | 1023_0 |"
    ) in report
    assert "| gram_01 | C2H4 | 6 | 100.00 | >=100.00 | 0.012345 |" in report
    assert "![Total-energy stability](total_energy_stability.png)" in report
    assert "![Wall time per simulated femtosecond](wall_time_per_fs.png)" in report


def test_energy_and_timing_plots_are_written_for_nine_systems(tmp_path):
    results = [
        {
            "system": f"gram_{index:02d}",
            "formula": "C2H4",
            "atoms": 6,
            "maximum_absolute_drift_meV_per_atom": 0.01 * index,
            "stable_duration_at_1meV_per_atom_fs": 100.0,
            "stable_through_full_run": True,
            "md_wall_seconds_per_simulated_fs": 200.0 + index,
            "total_wall_seconds_per_simulated_fs": 210.0 + index,
        }
        for index in range(1, 10)
    ]
    series = [
        {
            "time_fs": np.asarray([0.0, 50.0, 100.0]),
            "drift_meV_per_atom": np.asarray([0.0, 0.001 * index, -0.002 * index]),
        }
        for index in range(1, 10)
    ]
    energy_plot = tmp_path / "energy.png"
    timing_plot = tmp_path / "timing.png"

    plot_energy(results, series, energy_plot)
    plot_timing(results, timing_plot)

    assert energy_plot.stat().st_size > 0
    assert timing_plot.stat().st_size > 0
