# Scientific Performance Diagnostics

## Purpose

This directory contains focused experiments for reference convergence, final-orbital canonicalization, DFT force oracles, and relaxed-descriptor finite differences.

Every command uses the current source tree, frozen molecular inputs from `validation/scientific_performance/`, and thread limits applied before importing NumPy and PySCF. Generated JSON artifacts are written under `diagnostics/scientific_performance/results/`.

## DFT zero-correction oracle

`zero_gradient.py` compares the zero-correction DeePHF direct gradient with both the PySCF default fixed-grid gradient and the full finite-grid gradient produced with `grid_response=True`. The full-grid comparison is the valid RKS and UKS oracle because it differentiates the same energy surface.

```bash
uv run python diagnostics/scientific_performance/zero_gradient.py --workload S1-6-31G --family rks
uv run python diagnostics/scientific_performance/zero_gradient.py --workload S3-def2-SVP --family uks
```

## Displaced-reference validation

`displacement_scan.py` records every attempted point incrementally and distinguishes SCF execution, SCF convergence, state continuity, and strict method-validation failures. Converged references are canonicalized with their final-density Fock matrix before strict validation.

Recheck the S2/RKS convergence-boundary point with the current default controls:

```bash
uv run python diagnostics/scientific_performance/displacement_scan.py --workload S2-def2-TZVP --family rks --scope point --kind component --step 3e-4 --atom 3 --axis y --sign 1
```

Recheck the L3/UKS final-orbital canonicalization point:

```bash
uv run python diagnostics/scientific_performance/displacement_scan.py --workload L3-def2-SVP --family uks --scope point --kind component --step 3e-4 --atom 0 --axis x --sign 1
```

## Relaxed-descriptor finite difference

`descriptor_point.py` compares one analytic relaxed-descriptor coordinate with fresh-reference central differences at selected steps. It records explicit and response descriptor contributions, energy-gradient agreement, state continuity, and the worst descriptor atom and feature. `--conv-tol`, `--conv-tol-grad`, and `--max-cycle` can override workload SCF controls for a focused experiment.

```bash
uv run python diagnostics/scientific_performance/descriptor_point.py --workload L3-def2-SVP --family uhf --atom 11 --axis z --steps 3e-3 2e-3
```

## Execution order

1. Reproduce the DFT same-surface zero-correction invariant after a DFT gradient change.
2. Replay an exact displaced point after a reference-construction or SCF-control change.
3. Run the L3/UHF descriptor point experiment after an unrestricted response change.
4. Run the focused reference-finalization test, the affected analytic-force objective groups, and the complete suite.
