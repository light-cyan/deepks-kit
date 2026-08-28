# Published DeePHF B3LYP molecular dynamics

The workflow runs NVE trajectories with the published B3LYP(GRAM and Transition1x) correction network, GPU4PySCF B3LYP5/def2-TZVP references, the complete GPU coupled-perturbed density response, and analytic DeePHF nuclear forces. `B3LYP5` is the explicit LibXC spelling of the B3LYP definition used by the publication's PySCF 2.2.1 environment. Every electronic calculation must run inside a Slurm GPU allocation.

Prepare the audited checkpoint and nine neutral singlet GRAM test configurations from the Zenodo archive:

```bash
uv run python scripts/prepare_zenodo_deephf.py project.tar.gz deephf_b3lyp_assets
```

Validate the converted model against the archived descriptors and correction labels, reproduce one archived B3LYP baseline and descriptor, and compare one analytic gradient component with a central finite difference:

```bash
mkdir -p deephf_b3lyp_validation
sbatch --gres=gpu:1 --output=deephf_b3lyp_validation/slurm-%j.out --wrap="uv run python scripts/validate_published_deephf.py deephf_b3lyp_assets --output deephf_b3lyp_validation/validation.json"
```

Run the nine systems one at a time on a single available GPU. The command below reproduces the current protocol: 400 steps, a 0.25 fs timestep, 100 fs total duration, and a 100 K Maxwell-Boltzmann target temperature. The finite-system frame-0 temperature is a deterministic sample from that distribution and is reported separately.

```bash
sbatch --array=0-8%1 examples/deephf_md/run_slurm.sh deephf_b3lyp_assets/systems deephf_b3lyp_assets/b3lyp_gram_t1x.pth def2-tzvp deephf_b3lyp_md 400 0.25 100 20260821
```

Each trajectory contains `trajectory.traj`, `energy.csv`, and `summary.json`. The summary records total-energy drift, wall time per simulated femtosecond, the B3LYP grid controls, Slurm identifiers, and the accepted electronic-root lineage.

Use the original perturbative DeePHF numerical-force construction as a control by selecting `central_finite_difference`. This evaluates the complete `e_base + e_corr` energy at positive and negative Cartesian displacements of `1e-4` Bohr. A frame for an `N`-atom system requires `1 + 6N` complete DeePHF energy evaluations.

```bash
sbatch --array=0-8%1 examples/deephf_md/run_slurm.sh deephf_b3lyp_assets/systems deephf_b3lyp_assets/b3lyp_gram_t1x.pth def2-tzvp deephf_b3lyp_md_numerical 400 0.25 100 20260821 central_finite_difference 1e-4
```

Analyze the nine completed trajectories and generate the total-energy conservation and timing plots:

```bash
uv run python scripts/analyze_energy_stability.py deephf_b3lyp_md
```

The energy plot contains only the total-energy deviation from the initial NVE value. The reported stable duration ends at the first frame whose absolute total-energy drift exceeds 1 meV per atom.

Compare completed analytic and numerical-force campaigns on their common time grid:

```bash
uv run python scripts/compare_force_methods.py deephf_b3lyp_md deephf_b3lyp_md_numerical deephf_force_comparison
```
