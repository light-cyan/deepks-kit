# GPU DeePHF molecular dynamics

The DeePHF ASE calculator evaluates the GPU4PySCF reference, correction network, complete direct density response, and analytic nuclear gradient within one Slurm GPU allocation. The calculator rejects periodic systems, atomic-number changes, SCF failures, occupation changes, and discontinuous occupied subspaces.

Run one trajectory from an XYZ file whose comment line contains `charge` and `multiplicity`:

```bash
sbatch --gres=gpu:1 --wrap="uv run python scripts/run_ase_md.py sample.xyz --model model.pth --basis ccpvdz --output-directory md-output"
```

Run the nine independent samples from the supplied archive as a Slurm array:

```bash
sbatch --array=0-8 examples/deephf_md/run_slurm.sh random_cluster_samples_20260821.zip model.pth ccpvdz md-output
```

The first argument may also be an extracted sample directory. Optional trailing arguments set the step count, timestep in femtoseconds, temperature in kelvin, and random-velocity seed:

```bash
sbatch --array=0-2 examples/deephf_md/run_slurm.sh random_cluster_samples_20260821/samples/small NONE sto-3g md-output 100 0.1 100 20260821
```

Use `NONE` as the model argument for a native-reference validation trajectory. Every output directory contains `trajectory.traj`, `energy.csv`, and `summary.json`. The summary reports maximum and linear total-energy drift, MD and end-to-end wall time per simulated femtosecond, Slurm job identifiers, and the accepted electronic-root lineage.

Analyze nine completed trajectories using a `1 meV/atom` total-energy stability band and generate energy-conservation and timing plots:

```bash
uv run python scripts/analyze_energy_stability.py md-output
```

The ASE adapter uses the existing `CorrNet` correction model and its analytic descriptor derivative; it does not require an ASE-specific network. A checkpoint is reusable when its projector basis, descriptor width, element configuration, reference family, and training target match the simulated system. The supplied archive contains geometries and single-point metadata but no correction-model checkpoint or force labels, so it cannot by itself provide a new transferable correction model.

The MD entry point first runs GPU DIIS SCF and uses the GPU4PySCF Newton solver when DIIS does not converge. The default MD thresholds are recorded in `summary.json`; adjust the `--scf-*` options for harder electronic states.
