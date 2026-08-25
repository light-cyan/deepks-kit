# Partial Scientific Performance Campaign Report

## Status

Campaign `20260824-c62c392-final4` was terminated at the user's request. Termination was confirmed by process inspection at `2026-08-25T00:59:08+08:00`, with no remaining `run_campaign.py` or `worker.py` process.

This is a partial evidence report. It records completed work but does not assign an overall scientific-performance verdict because the scientific phase is incomplete and the performance, cross-revision, and aggregation phases did not run.

## Evidence Identity

| Field | Value |
| --- | --- |
| Campaign ID | `20260824-c62c392-final4` |
| Campaign creation | `2026-08-24T13:39:19.553994+00:00` |
| Current revision | `c62c3929cdb08309acbbe8ac45fddd5f00975202` |
| Validation driver revision | `c62c3929cdb08309acbbe8ac45fddd5f00975202` |
| Config hash | `c36954c32d4ed9f5a094bb91df2c62de26a809fe260ba647c07eaca86d132b2e` |
| Environment lock hash | `62d6409e684747b95d1593e538961f758b2d89b7cb88dfcd4584d469735f4ee8` |
| Validation hash | `7d0622745975e9fd21cc8b3352e1e87fa2f97362a66ee026e2cfdc2336a4ace9` |
| Completed profile | `deterministic-1t` with one declared thread and CPU affinity to core 0 |
| Evidence root | `validation/scientific_performance/runs/20260824-c62c392-final4/` |
| Preserved archive | `validation/scientific_performance/reports/raw/20260824-c62c392-final4-partial.tar.gz` |
| Archive SHA-256 | `5e6d1616c7b02ac2a719b366c1f5707a9df522c77419f3e6ef3babbfb8b62492` |

All 23 result records carry the same validation hash. The production source worktree recorded in the result environment was tracked-clean; the campaign manifest recorded a tracked modification to `validation/scientific_performance/README.md` in the validation driver tree.

The recorded environment used an AMD EPYC 7K62 processor, Python 3.11.15, PySCF 2.14.0, NumPy 2.4.6, SciPy 1.17.1, Torch 2.13.0+cpu, and LibXC 7.0.0.

## Completion Summary

| Scope | Result |
| --- | --- |
| Full expanded campaign | 391 child cases |
| Result records written | 23 of 391, or 5.9% |
| Successful child exits | 22 |
| Unsuccessful child exits | 1 |
| Interrupted child without a result record | 1, `scientific/S3-def2-SVP/UKS` |
| Child cases not started | 367 |
| Sum of recorded child elapsed time | 3:16:07.53 |
| Maximum observed peak RSS | 1,493,172 KiB, or 1.424 GiB |
| Longest completed child | `preflight/X1-def2-TZVP/RHF`, 1:51:27 |

Verification, setup, and all 15 preflight cases completed. The scientific phase produced five result records, the next scientific case was interrupted, and the remaining campaign phases did not start.

Across the 23 result records, integrity was true for 22 and unavailable for the child that exited unsuccessfully; resource status was true for 22 and false for that unsuccessful child; scientific status was true for 18, false for 2, and not applicable for 3 verification/setup records; performance status was unavailable for every record.

## Verification and Setup

| Action | Outcome | Test time or child elapsed | Peak RSS KiB |
| --- | --- | ---: | ---: |
| Focused verification | 692 passed | 81.54 s pytest; 1:25.64 child | 429,068 |
| Complete verification | 812 passed | 84.63 s pytest; 1:27.61 child | 431,876 |
| Deterministic checkpoint setup | Passed | 0:01.68 | 304,900 |

## Preflight Results

All 15 preflight cases passed scientific, integrity, and resource acceptance without a timeout. The largest direct-versus-Z-vector maximum absolute difference was `5.573e-12`, below the `1e-8` threshold. The smallest correction-gradient response signal was `2.601e-3` and the smallest descriptor response signal was `2.863e-1`, both above the `1e-4` anti-vacuity thresholds.

| Workload | Family | AO | Response dimension | Direct/Z max abs | Response signal | Descriptor signal | Elapsed | Peak RSS KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1-def2-SVP | RHF | 96 | 1,520 | 2.903e-14 | 1.283e-2 | 4.313e-1 | 2:10.67 | 488,684 |
| L1-def2-SVP | RKS | 96 | 1,520 | 3.229e-14 | 1.174e-2 | 4.120e-1 | 2:41.43 | 587,576 |
| L1-def2-TZVP | RHF | 172 | 3,040 | 2.550e-14 | 1.227e-2 | 3.999e-1 | 16:08.71 | 1,334,580 |
| L1-def2-TZVP | RKS | 172 | 3,040 | 1.130e-14 | 1.198e-2 | 3.721e-1 | 15:37.22 | 1,493,172 |
| L2-def2-SVP | RHF | 114 | 1,953 | 2.899e-13 | 2.601e-3 | 2.907e-1 | 4:18.39 | 585,116 |
| L2-def2-SVP | RKS | 114 | 1,953 | 3.950e-14 | 4.196e-3 | 2.863e-1 | 4:51.61 | 693,460 |
| L3-def2-SVP | UHF | 123 | 4,826 | 5.573e-12 | 8.228e-3 | 4.967e-1 | 10:34.62 | 704,140 |
| L3-def2-SVP | UKS | 123 | 4,826 | 4.846e-13 | 7.974e-3 | 3.394e-1 | 14:58.70 | 912,352 |
| S1-6-31G | RHF | 22 | 112 | 9.068e-15 | 3.959e-3 | 3.703e-1 | 0:03.45 | 368,856 |
| S1-6-31G | RKS | 22 | 112 | 1.853e-14 | 3.982e-3 | 3.729e-1 | 0:04.06 | 378,324 |
| S2-def2-TZVP | RHF | 74 | 528 | 8.174e-15 | 4.047e-3 | 3.980e-1 | 0:25.35 | 407,072 |
| S2-def2-TZVP | RKS | 74 | 528 | 1.323e-13 | 3.423e-3 | 3.432e-1 | 0:26.95 | 433,776 |
| S3-def2-SVP | UHF | 43 | 586 | 3.731e-14 | 3.715e-3 | 2.994e-1 | 0:08.66 | 376,424 |
| S3-def2-SVP | UKS | 43 | 586 | 3.723e-14 | 3.438e-3 | 2.901e-1 | 0:17.62 | 409,564 |
| X1-def2-TZVP | RHF | 258 | 6,840 | 1.215e-13 | 1.118e-2 | 4.054e-1 | 1:51:27 | 726,148 |

## Scientific Results

The scientific phase contains 14 expected cases. Five cases wrote result records, one additional case was interrupted without a result record, and eight cases did not start.

| Workload | Family | Status | Direct/Z max abs | FD component max | FD descriptor max | FD direction max | Zero-gradient max abs | Elapsed | Peak RSS KiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1-6-31G | RHF | Pass | 9.068e-15 | 4.141e-7 | 1.988e-7 | 2.460e-7 | 0 | 0:28.34 | 373,236 |
| S1-6-31G | RKS | Scientific failure: `zero_gradient` | 1.853e-14 | 3.492e-7 | 1.058e-6 | 2.519e-7 | 4.707e-3 | 0:51.20 | 379,288 |
| S2-def2-TZVP | RHF | Pass | 8.174e-15 | 3.640e-8 | 1.389e-7 | 3.022e-8 | 0 | 3:46.81 | 438,448 |
| S2-def2-TZVP | RKS | Execution failure: displaced native RKS reference did not converge | — | — | — | — | — | 2:17.18 | 443,872 |
| S3-def2-SVP | UHF | Pass | 3.731e-14 | 2.402e-8 | 1.490e-7 | 1.331e-8 | 0 | 1:34.63 | 382,264 |
| S3-def2-SVP | UKS | Interrupted by user request; no result record | — | — | — | — | — | — | — |

The S1 RKS child exited successfully and passed direct/Z-vector agreement, compact/detailed agreement, checkpoint identity, repeat identity, anti-vacuity, zero-energy, and all finite-difference checks. Its only failed acceptance check was the zero-correction gradient comparison: the maximum difference from the native gradient was `4.707485055421401e-3` Hartree/bohr against a `1e-9` threshold.

The S2 RKS central preflight passed, but its scientific finite-difference workflow exited while constructing a fresh displaced RKS reference because that SCF reference did not converge. The result contains no completed central scientific metrics.

The S3 UKS process had begun and emitted only the environment warning in `stderr.log` when termination was requested. It did not write `result.json`, so no acceptance category is assigned.

## Unfinished Scope

No result record was produced for the remaining eight large scientific cases or for the dense, conditioning, benchmark, selection, invariance, DFT, scanner, force-data, cross-revision, and aggregate phases. Consequently, the campaign provides no throughput comparison, timing regression decision, current-versus-historical comparison, data-pipeline verdict, or overall campaign verdict.
