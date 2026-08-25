# Current Repair Scientific Run

## Provenance

This run executed the current working tree directly rather than a detached clean `HEAD` worktree. The base revision is `0b1405fbc556376cfce833986b5cc71bd3c7a3f6`, the tracked diff SHA-256 is `f738d92af3e1100f57d9b82cbd0fa9770f108f61c2caa98c9aee9d38d000a46d`, and the validation input and script hash is `f0a11922ba8bed1155ae9786f540d75662f3efbf4910e1636ca74fbab64e13a9`.

All fourteen configured scientific cases ran under the `deterministic-1t` profile. Every child completed without a timeout or process failure.

## Complete scientific matrix

| Workload | Family | Scientific | Force FD | Direction FD | Descriptor FD | Direct/Z-vector | Worst descriptor point | Elapsed |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| L1-def2-SVP | RHF | FAIL | 3.167e-6 | 7.724e-7 | 4.762e-5 | 2.912e-14 | 3e-3, atom 0 x | 9:52.69 |
| L1-def2-SVP | RKS | PASS | 2.965e-6 | 7.346e-7 | 8.086e-6 | 1.845e-13 | 3e-3, atom 0 y | 11:34.02 |
| L1-def2-TZVP | RHF | FAIL | 2.147e-6 | 1.103e-6 | 1.092e-4 | 2.553e-14 | 3e-3, atom 0 x | 1:17:37 |
| L1-def2-TZVP | RKS | FAIL | 2.342e-6 | 7.966e-7 | 1.530e-5 | 1.127e-14 | 3e-3, atom 0 x | 1:00:25 |
| L2-def2-SVP | RHF | FAIL | 1.942e-6 | 3.158e-7 | 1.732e-5 | 1.153e-12 | 3e-3, atom 2 x | 20:42.89 |
| L2-def2-SVP | RKS | FAIL | 1.997e-6 | 3.199e-7 | 1.699e-5 | 6.242e-14 | 3e-3, atom 0 x | 19:51.40 |
| L3-def2-SVP | UHF | FAIL | 2.560e-6 | 4.361e-7 | 1.332e-5 | 5.517e-14 | 3e-3, atom 7 y | 45:52.14 |
| L3-def2-SVP | UKS | FAIL | 2.943e-6 | 3.356e-7 | 4.548e-5 | 4.851e-13 | 2e-3, atom 2 y | 44:00.45 |
| S1-6-31G | RHF | PASS | 4.145e-7 | 2.460e-7 | 1.993e-7 | 9.067e-15 | 1e-3, atom 1 x | 0:26.80 |
| S1-6-31G | RKS | PASS | 3.492e-7 | 2.520e-7 | 2.328e-7 | 1.854e-14 | 3e-4, atom 0 y | 0:48.89 |
| S2-def2-TZVP | RHF | PASS | 3.497e-6 | 2.903e-6 | 1.455e-6 | 8.188e-15 | 3e-3, atom 1 x | 3:07.10 |
| S2-def2-TZVP | RKS | PASS | 2.970e-6 | 3.051e-6 | 1.453e-6 | 1.213e-13 | 3e-3, atom 3 y | 2:20.82 |
| S3-def2-SVP | UHF | PASS | 2.437e-6 | 1.337e-6 | 3.526e-6 | 3.730e-14 | 3e-3, atom 1 x | 0:59.38 |
| S3-def2-SVP | UKS | FAIL | 2.542e-6 | 1.513e-6 | 4.551e-5 | 4.417e-14 | 3e-3, atom 1 z | 1:34.14 |

Six cases pass every scientific check and eight cases fail only the relaxed-descriptor finite-difference check. All fourteen integrity checks pass. The maximum complete-energy component error is `3.497e-6 Ha/Bohr`, the maximum directional error is `3.051e-6 Ha/Bohr`, and the maximum direct-versus-Z-vector error is `1.153e-12 Ha/Bohr`; all are inside their configured thresholds.

## Step sweeps

The S3/UKS atom 1 z descriptor error follows central-difference truncation behavior under the default SCF controls: `4.551e-5` at `3e-3 Bohr`, `2.025e-5` at `2e-3 Bohr`, `5.066e-6` at `1e-3 Bohr`, `1.267e-6` at `5e-4 Bohr`, `4.508e-7` at `3e-4 Bohr`, and `5.164e-8` at `1e-4 Bohr`.

The L1/def2-SVP RHF atom 0 x descriptor error also follows truncation behavior: `4.762e-5` at `3e-3 Bohr`, `2.111e-5` at `2e-3 Bohr`, `5.267e-6` at `1e-3 Bohr`, `1.316e-6` at `5e-4 Bohr`, `4.741e-7` at `3e-4 Bohr`, and `5.269e-8` at `1e-4 Bohr`.

The L3/UHF atom 11 z descriptor error is SCF-noise limited and nonmonotonic under the default controls. The error is inside `1e-5 Bohr^-1` from `2e-3` through `4e-3 Bohr`, reaches `8.185e-7` at `3e-3 Bohr`, and exceeds the threshold at `1.5e-3 Bohr` and below. A central reference with `conv_tol_grad=1e-8` does not converge under the configured standard SCF procedure.

The L3/UKS atom 2 y descriptor error is also SCF-noise limited under the default `conv_tol_grad=1e-7`: `3.036e-5` at `3e-3 Bohr`, `4.548e-5` at `2e-3 Bohr`, `8.498e-5` at `1e-3 Bohr`, and `3.019e-4` at `3e-4 Bohr`. Tightening `conv_tol_grad` to `1e-8` reduces the corresponding errors to `5.985e-6`, `4.086e-6`, `9.505e-6`, and `2.602e-5`, respectively. The successful `1e-3` through `3e-3 Bohr` strict-SCF points support the analytic descriptor Jacobian and identify displaced-reference precision as the dominant default-control error source.

## Finding

Final-density orbital canonicalization fixes the observed L3/UKS strict-reference rejection and leaves all complete-energy force checks accurate. The configured global `3e-3` and `2e-3 Bohr` long-workload steps do not form a valid common descriptor-validation range. Ordinary S3 and L1 points require smaller steps to control truncation error, L3/UHF requires larger steps because tighter standard UHF references do not converge, and L3/UKS requires tighter SCF controls to expose a usable finite-difference range.
