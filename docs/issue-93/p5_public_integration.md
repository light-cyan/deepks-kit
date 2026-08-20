# P5 Public Integration and Release Contract

## 1. Status

P5 integrates the accepted Issue #93 scientific backends through one strict public Python workflow and the `deepks deephf` command.

## 2. Public construction and dispatch

`deepks.deephf.make_deephf(reference, model, ...)` dispatches exact native RHF, UHF, RKS, and UKS references to `DeePHF`, `UHFDeePHF`, `RKSDeePHF`, and `UKSDeePHF`. Each concrete constructor immediately enforces its scientific support contract.

`deepks.deephf.build_reference(molecule, family, ...)` builds a fresh exact native reference with strict convergence controls. RKS and UKS construction applies the characterized normalized pure-LDA functional and deterministic finite grid before SCF.

`deepks.deephf.evaluate_molecule(...)` returns canonical `converged`, `e_base`, `e_corr`, `e_tot`, `descriptor`, `gradient`, and `force` results. `force` is exactly `-gradient`, energies use `Eh`, coordinates use `Bohr`, and gradients and forces use `Eh/Bohr`.

## 3. CLI workflow

`deepks deephf INPUT.yaml` reads molecular systems through the method-neutral data input layer, loads an optional strict CorrNet checkpoint, constructs the selected native reference family, selects `direct` or `zvector`, evaluates every frame, and writes one `.npy` file per canonical output.

The runnable [RHF inference configuration](../../examples/deephf/rhf_inference.yaml) uses a zero correction and therefore reproduces the native RHF energy and gradient through the complete direct backend.

## 4. Compatibility and lifecycle

Direct remains the default backend for every reference family. Scalar-adjoint inference is explicitly selected and remains distinct from model-independent direct density response and RHF relaxed-force data generation.

The strict RHF force-training schema and checkpoint contract remain unchanged. Public inference loads an energy-only checkpoint with exact state keys and validates its projector, double precision, deterministic scalar output, complete model sensitivity, descriptor differentiability, and reference state at the force boundary.

The package architecture test assigns each PySCF response, gradient, LibXC, grid, and private-state facility to its compatibility adapter and prevents direct-response symbols from entering scalar-adjoint drivers.

## 5. Release verification

```bash
uv sync --locked --python 3.11
uv run pytest -q
uv build
git diff --check
```

Final release verification completes 838 tests under Python 3.11, exact locked dependency synchronization, bytecode compilation, source and wheel builds containing the UKS adapters, public workflow, documentation, examples and tests, CLI help and runnable RHF inference, architecture ownership checks, and a clean whitespace audit.
