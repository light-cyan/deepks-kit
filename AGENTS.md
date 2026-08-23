# Repository guide

## Current objective

- Upstream [`deepmodeling/deepks-kit` Issue #93](https://github.com/deepmodeling/deepks-kit/issues/93), covering exact analytic DeePHF nuclear forces and force-aware training, is complete in this fork.
- Current development focuses on improving computational efficiency, reducing redundancy in code and execution paths, and fixing correctness defects.

## Environment

- This is a Python 3.10+ project. Python 3.11 is the standard development and test version.
- `uv` manages the environment and dependencies. Project metadata is in `pyproject.toml`, and exact dependency versions are in `uv.lock`.
- Create or refresh the environment with `uv sync --python 3.11` and run project commands through `uv run`.
- Use `uv sync --locked --python 3.11` for verification and CI-like workflows.

## Project layout

- `deepks/model/`: neural-network models, data readers, training, and model evaluation.
- `deepks/deephf/`: perturbative DeePHF inference, response calculations, analytic nuclear gradients, and force-data generation.
- `deepks/scf/`: PySCF integration, self-consistent calculations, and analytic nuclear gradients.
- `deepks/iterate/` and `deepks/task/`: iterative workflows and task execution.
- `examples/`: runnable configurations and sample data.
- `tests/baseline/`: smoke checks for currently supported core functionality.
- `docs/legacy/` contains archived material from past work. It is not a current requirement or development input and does not need to be read unless the user explicitly requests historical comparison.

## Maintenance priorities

- Prefer simpler production paths that avoid redundant validation, repeated calculations, unnecessary materialization, and duplicate state.
- Preserve scientific correctness and public behavior with focused regression tests.
- Tests that enforce obsolete implementation details or unnecessary computations may be removed or replaced when those implementation paths are retired.

## Creating tests

- Organize test modules by their testing objective under `tests/<objective>/`.
- Place core-function smoke checks in `tests/baseline/`.
- Create a dedicated objective directory for each substantial development effort.

## Test quality

- Tests are self-contained and write temporary data only to pytest-provided temporary directories.
- Keep numerical tests deterministic. Set random seeds when exact values or reproducibility matter, and use `numpy.testing` or `torch.testing` with explicit tolerances for floating-point comparisons.
- For analytic-force changes, test finite values and shapes and, when practical, compare analytic gradients against finite differences on a small molecule.
- Prefer small molecules, minimal basis sets, and compact networks so the full suite remains suitable for local development.

## Verification

Run the relevant objective directory first, followed by the complete suite:

```bash
uv run pytest tests/baseline
uv run pytest
```

For dependency or packaging changes, also run:

```bash
uv sync --locked --python 3.11
uv build
```
