# DeePKS-kit

DeePKS-kit is a machine-learning toolkit for constructing quantum-chemistry energy corrections from projected one-particle density descriptors. It supports both self-consistent DeePKS calculations and perturbative DeePHF calculations around converged native PySCF references.

The current implementation provides strict analytic molecular gradients, scalar-adjoint inference, force-aware RHF training, command-line workflows, and Python APIs. Scientific force paths validate the reference, model, projector, descriptor differentiability, response equations, numerical conditioning, and persisted provenance before publishing a result.

## Features

### DeePHF energy and analytic-gradient inference

The public DeePHF workflow supports four native PySCF reference families:

| Reference | Supported tier | Direct analytic gradient | Z-vector analytic gradient |
| --- | --- | --- | --- |
| RHF | Converged closed-shell molecular RHF | Yes | Yes |
| UHF | Converged open-shell molecular UHF | Yes | Yes |
| RKS | Converged closed-shell pure-LDA RKS | Yes | Yes |
| UKS | Converged open-shell pure-LDA UKS | Yes | Yes |

The `direct` backend constructs the complete coordinate-wise first-order density and relaxed descriptor response. The `zvector` backend solves one correction-specific transpose response equation for each scalar correction energy. Both backends include occupied-virtual response and AO-metric contributions, and the RKS and UKS backends also include finite-grid coordinate and weight response.

The accepted RKS and UKS tier uses normalized `LDA_X + LDA_C_VWN`, native LibXC 7.0.0, and the characterized deterministic unpruned atom-centered grid configured by the public workflow.

The public inference result contains:

- `converged`: native-reference convergence state.
- `e_base`: native PySCF reference energy in `Eh`.
- `e_corr`: learned correction energy in `Eh`.
- `e_tot`: `e_base + e_corr` in `Eh`.
- `descriptor`: projected density descriptor.
- `gradient`: complete analytic nuclear gradient in `Eh/Bohr`.
- `force`: `-gradient` in `Eh/Bohr`.

### RHF relaxed-force data and training

The strict RHF force-training workflow generates model-independent direct-response data and trains a correction model against both energy and force targets.

The persisted contract includes the canonical fields `e_corr_target`, `f_corr_target`, and `dq_dR_relaxed`, together with array hashes, projector identity, reference state, response diagnostics, units, axis conventions, and compatibility fingerprints. Training, validation, saved-data testing, restart, and checkpoint loading verify this contract before force contraction.

The force prediction uses the complete model sensitivity and relaxed descriptor Jacobian:

```text
f_corr = -sum_I,k (d e_corr / d q[I,k]) (d q[I,k] / d R)
```

### Self-consistent DeePKS

The self-consistent DeePKS package runs RKS/UKS SCF and analytic gradients through GPU4PySCF. Projected-density descriptors, neural corrections, training, and saved-data evaluation run on CUDA through PyTorch, with DLPack transfers between CuPy and PyTorch at the SCF boundary.

### Model and data utilities

- `CorrNet` provides double-precision projected-density correction models with linear, dense, residual, normalization, and checkpoint facilities.
- Readers support energy-only datasets and the strict RHF relaxed-force schema.
- Training reports separate energy and force metrics and supports validated force-aware restart checkpoints.
- Saved-data testing evaluates compatible energy and force datasets.
- Iteration and task utilities stage local or SSH-backed workflows and submit every GPU calculation through Slurm.

## Command-line interface

The installed `deepks` command, also available as `dks`, provides these subcommands:

| Command | Purpose |
| --- | --- |
| `deepks deephf` | Run strict perturbative energy and analytic-gradient inference. |
| `deepks train` | Train or restart a correction model. |
| `deepks test` | Evaluate a trained model on saved data. |
| `deepks scf` | Run self-consistent DeePKS calculations. |
| `deepks stats` | Collect statistics from calculation results. |
| `deepks iterate` | Run iterative data-generation, training, and SCF workflows. |

Use the built-in help for the current options:

```bash
uv run deepks --help
uv run deepks deephf --help
uv run deepks train --help
```

## Requirements and dependencies

### Runtime requirements

- Python 3.10 or newer; Python 3.11 is the standard development and verification version.
- Linux on an NVIDIA GPU with CUDA 12.8 compatibility and compute capability 7.0 or newer.
- Slurm with GPU generic resources enabled.
- Sufficient memory for dense response operators in the selected strict support tier.

### Direct dependencies

| Dependency | Purpose | Current locked version |
| --- | --- | --- |
| NumPy | Arrays, linear algebra, descriptors, response tensors, and persisted data | 2.2.6 |
| GPU4PySCF CUDA 12 | GPU SCF, density-functional integration, and analytic gradients | 1.8.1 |
| PySCF | Molecular definitions and response primitives used by GPU4PySCF and strict DeePHF | 2.14.0 |
| PyTorch CUDA 12.8 | GPU descriptors, CorrNet models, automatic differentiation, training, and checkpoint state | 2.11.0+cu128 |
| CuPy CUDA 12 | GPU array storage and GPU4PySCF interoperability | 13.4.1 |
| cuTENSOR CUDA 12 | Accelerated tensor contractions | 2.2.0 |
| ruamel.yaml | YAML configuration loading | 0.19.1 |
| Paramiko | SSH-backed task execution | 5.0.0 |

GPU execution is mandatory. Runtime entry points reject CPU devices, unavailable CUDA runtimes, and execution outside a Slurm allocation instead of falling back to the CPU. The runtime preloads the locked CUDA 12.8 runtime and NVRTC libraries before CuPy initialization so Blackwell devices compile GPU kernels against the matching toolkit.

### Development and build dependencies

- `pytest` 9.1.1 is included in the development dependency group.
- `setuptools` and `setuptools-scm` provide source and wheel builds and derive the package version from Git metadata.
- `uv` manages the virtual environment, dependency lock, commands, and builds.

## Installation

Install the locked development environment from a repository checkout:

```bash
uv sync --locked --python 3.11
uv run deepks --help
```

Submit calculations with one visible GPU, for example:

```bash
srun --gres=gpu:1 uv run deepks scf INPUT.yaml
srun --gres=gpu:1 uv run deepks train INPUT.yaml
```

Build a source distribution and wheel:

```bash
uv build
```

The project metadata declares the console scripts `deepks` and `dks`.

## Quick start: strict DeePHF inference

The repository includes a runnable zero-correction RHF example:

```bash
uv run deepks deephf examples/deephf/rhf_inference.yaml
```

Its configuration is:

```yaml
systems:
  - examples/deephf/h2.xyz
reference: rhf
model_file: NONE
basis: sto-3g
projector_basis:
  - [0, [0.8, 1.0]]
backend: direct
dump_dir: examples/deephf/inference_output
mol_args:
  unit: Angstrom
verbose: 0
```

`model_file: NONE` selects a zero correction and reproduces the native reference energy and gradient through the complete backend. A trained CorrNet checkpoint path enables learned correction inference.

The workflow writes one NumPy file per canonical output under a system-specific output directory:

```text
converged.npy
e_base.npy
e_corr.npy
e_tot.npy
descriptor.npy
gradient.npy
force.npy
```

Select a reference and gradient backend in YAML or on the command line:

```bash
uv run deepks deephf input.yaml --reference uhf --backend direct
uv run deepks deephf input.yaml --reference rks --backend zvector
uv run deepks deephf input.yaml --reference uks --backend zvector
```

The workflow constructs a fresh native reference for each frame, applies strict convergence controls, validates the requested method tier, and fails without returning a partial or fallback gradient when a force contract is not satisfied.

## Python API

The public API can construct references, dispatch method classes, and evaluate complete results:

```python
from pyscf import gto

from deepks.deephf import build_reference, evaluate_molecule, make_deephf

molecule = gto.M(
    atom="H 0 0 0; H 0 0 0.7408480953",
    basis="sto-3g",
    unit="Angstrom",
    spin=0,
    symmetry=False,
    cart=False,
    verbose=0,
)
projector_basis = [[0, [0.8, 1.0]]]

reference = build_reference(molecule, "rhf")
method = make_deephf(reference, None, projector_basis=projector_basis)
energy = method.kernel()
direct_gradient = method.gradient(backend="direct")
zvector_gradient = method.gradient(backend="zvector")

result = evaluate_molecule(
    molecule,
    None,
    family="rhf",
    backend="direct",
    projector_basis=projector_basis,
)
```

The concrete method classes are `DeePHF`, `UHFDeePHF`, `RKSDeePHF`, and `UKSDeePHF`. Their direct response objects and scalar-adjoint objects expose immutable diagnostic partitions for scientific validation.

For a multi-frame system, `deepks deephf` carries the preceding accepted density into the next SCF calculation and rejects a frame whose occupied-subspace overlap falls below `root_overlap_tolerance` (default `0.5`). Each successful system output contains `deephf_provenance.json` with the state lineage and spin-resolved overlap diagnostics; a force-trained checkpoint also contributes its target identity.

## Quick start: RHF energy-and-force training

Generate the deterministic example datasets and train the model:

```bash
uv run python examples/deephf/generate_rhf_force_data.py
uv run deepks train examples/deephf/rhf_force_train.yaml
```

The generator creates strict training and validation datasets under `examples/deephf/data`. The training configuration selects `force_mode: deephf_relaxed`, `f_corr_target`, and `dq_dR_relaxed`, trains in double precision, and writes `examples/deephf/model.pth`.

Application code can generate strict RHF force data through the public producer:

```python
from deepks.deephf import write_rhf_force_dataset

contract = write_rhf_force_dataset(
    output_directory,
    references,
    projector_basis=projector_basis,
    e_target=target_energies,
    f_target=target_forces,
    target={
        "method": "CCSD(T)",
        "basis": "cc-pVTZ",
        "software": "target-code",
        "version": "1.0",
        "frozen_core": True,
        "relativistic": "none",
        "state": "closed-shell singlet ground state",
        "energy_force_consistent": True,
        "settings": {"energy_tolerance": 1.0e-10},
    },
)
```

Target energies use `Eh`, target forces use `Eh/Bohr`, coordinates use `Bohr` internally, and `dq_dR_relaxed` uses `Bohr^-1` with the positive `dq/dR` convention. The required target identity is part of dataset compatibility and force-training checkpoint metadata, so grouped training rejects labels produced by a different target calculation.

## Self-consistent and iterative workflows

The existing self-consistent and iterative examples are organized under `examples/water_single`, `examples/water_cluster`, `examples/train_input`, and `examples/iterate`.

Typical entry points are:

```bash
srun --gres=gpu:1 uv run deepks scf INPUT.yaml
srun --gres=gpu:1 uv run deepks test INPUT.yaml
uv run deepks stats INPUT.yaml
uv run deepks iterate INPUT.yaml
```

Use each subcommand's `--help` output and the matching example configuration for its accepted arguments.

## Scientific support contract

Strict analytic-force inference requires an exact native molecular PySCF reference, a converged Aufbau state, complete real canonical orbitals, finite double-precision densities and model state, occupied and virtual spaces, a compatible fixed projector, a differentiable descriptor spectrum, and a stable, finite, well-conditioned response operator.

RHF and UHF use all-electron spherical molecular references with symmetry disabled. The RKS and UKS tiers additionally bind the normalized pure-LDA functional, LibXC version, NumInt settings, finite atom-centered grid, Becke partition, grid weights, and grid-response derivatives.

The direct and Z-vector backends independently check equation residuals and reconstruction identities. A response, adjoint, model, projector, descriptor, convergence, provenance, or numerical-conditioning failure raises an explicit error and does not fall back to a partial gradient.

Archived Issue #93 design material is stored under `docs/legacy/issue-93/` for historical reference.

## Project layout

| Path | Contents |
| --- | --- |
| `deepks/descriptor/` | Method-independent projected-density descriptors and derivatives. |
| `deepks/deepks/` | Self-consistent DeePKS method and gradients. |
| `deepks/deephf/` | Perturbative methods, response adapters, direct gradients, Z-vector drivers, force producer, scanner, and public workflow. |
| `deepks/data/` | Data fields, strict force schema, persistence, and statistics. |
| `deepks/model/` | CorrNet, readers, training, evaluation, and saved-data testing. |
| `deepks/iterate/` | Iterative workflow orchestration. |
| `deepks/task/` | Local and remote task execution. |
| `examples/` | Runnable inference, training, SCF, and iteration configurations. |
| `tests/` | Baseline, analytic-force, Z-vector, force-training, workflow, and architecture tests. |
| `docs/legacy/` | Archived material from past work. |

## Verification

Run the baseline tests first, followed by the complete suite:

```bash
uv run pytest tests/baseline
uv run pytest
```

Run the release-oriented checks with the locked Python 3.11 environment:

```bash
uv sync --locked --python 3.11
uv run pytest -q
uv build
git diff --check
```

The current accepted repository state completes 838 tests and builds both the source distribution and wheel.

## References

1. Y. Chen, L. Zhang, H. Wang, and W. E, “Ground State Energy Functional with Hartree–Fock Efficiency and Chemical Accuracy,” *The Journal of Physical Chemistry A* 124, 7155–7165 (2020).
2. Y. Chen, L. Zhang, H. Wang, and W. E, “DeePKS: A Comprehensive Data-Driven Approach toward Chemically Accurate Density Functional Theory,” *Journal of Chemical Theory and Computation* 17, 170–181 (2021).
