# Scientific Diagnostic Findings

## Scope

This report records the current-working-tree reference-finalization checks, the complete fourteen-case scientific matrix, and focused descriptor finite-difference step scans.

## Reference finalization

The public reference builder uses `conv_tol=1e-12` and canonicalizes occupied and virtual orbitals with the Fock matrix evaluated from the final density. The S2/def2-TZVP RKS boundary point converges under this energy tolerance. The L3/def2-SVP UKS atom 0, x, positive `3e-4 Bohr` point passes strict method validation after canonicalization, with maximum alpha and beta canonical residuals of `4.45883e-8` and `5.27570e-8`.

## Complete scientific matrix

All fourteen configured RHF, RKS, UHF, and UKS scientific cases completed in the current working tree. Six cases pass every scientific check and eight fail only the relaxed-descriptor finite-difference check. All integrity checks pass.

Across all fourteen cases, the maximum complete-energy component error is `3.497e-6 Ha/Bohr`, the maximum directional error is `3.051e-6 Ha/Bohr`, and the maximum direct-versus-Z-vector error is `1.153e-12 Ha/Bohr`. Zero-correction, compact-versus-detailed, repeated-input, checkpoint-reload, complete-energy finite-difference, and directional finite-difference checks pass in every case.

The relaxed-descriptor failures are S3/UKS, L1/def2-SVP RHF, both L1/def2-TZVP families, both L2/def2-SVP families, and both L3 families. Their maximum errors range from `1.332e-5` to `1.092e-4 Bohr^-1` against the `1e-5 Bohr^-1` threshold.

## Descriptor finite differences

S3/UKS and L1/RHF focused scans show second-order truncation error at `3e-3` and `2e-3 Bohr`; both pass at `1e-3 Bohr` and improve further at smaller steps. L2 shows the same step dependence in its complete results.

L3/UHF is SCF-noise limited at small steps under the default controls. Its focused atom 11 z scan passes from `2e-3` through `4e-3 Bohr` and fails at `1.5e-3 Bohr` and below. A `conv_tol_grad=1e-8` central UHF reference does not converge under the configured standard SCF procedure.

L3/UKS is SCF-noise limited and nonmonotonic under the default `conv_tol_grad=1e-7`. Tightening `conv_tol_grad` to `1e-8` reduces the focused atom 2 y descriptor errors at `3e-3`, `2e-3`, and `1e-3 Bohr` to `5.985e-6`, `4.086e-6`, and `9.505e-6`, respectively. These strict-SCF points pass and support the analytic descriptor Jacobian.

The current global `3e-3` and `2e-3 Bohr` long-workload steps are not a common stable range. The required finite-difference controls depend on workload and reference family.

## DFT force oracle

RKS and UKS zero-correction DeePHF gradients match PySCF gradients with `grid_response=True` in every completed scientific case.

## Regression verification

The focused reference-finalization regression passes, the baseline and four analytic-force objective groups pass 410 tests, and the complete suite passes 894 tests.

## Artifacts

The complete run summary is `runs/current_repair_20260825/SUMMARY.md`. Every scientific case contains `result.json`, `stdout.log`, `stderr.log`, `time.txt`, and `exit_status.txt`. Focused scans retain their full machine-readable step data under `runs/current_repair_20260825/focused/`.
