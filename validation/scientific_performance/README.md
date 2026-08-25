# Scientific Correctness and Performance Validation Plan

## Purpose

This plan defines long-running, high-stringency validation tasks for the current molecular DeePHF implementation. It evaluates scientific accuracy, execution time, peak memory, scaling behavior, repeated-work elimination, compact result paths, direct response, matrix-free Z-vector response, DFT validation transactions, atom subsets, coordinate blocking, scanners, force-data production, and force-aware model workflows.

The design was prepared against revision `59e0a71` after reviewing `docs/legacy/project-audits/CurrentProjectAudit.md` through `docs/legacy/project-audits/CurrentProjectAudit5.md`, the current response and adjoint implementations, the current compact direct and Z-vector drivers, the response-scalability tests, and the archived Issue #93 validation artifacts.

The archived validation under `validation/archive/issue_93_release/` remains a fixed record of the earlier dense-adjoint implementation and its release results. New runs must write only under this directory or an explicitly selected external output directory.

## Validation questions

The validation must answer the following questions with numerical evidence:

1. Do direct and Z-vector gradients remain scientifically equivalent for RHF, UHF, RKS, and UKS as molecular size, basis size, grid cost, and response dimension increase?
2. Does the current matrix-free GMRES adjoint reproduce an explicitly materialized dense response-operator solve on systems where a dense oracle is affordable?
3. Do complete analytic gradients agree with fresh-reference complete-energy finite differences rather than only with another analytic implementation?
4. Does compact execution reproduce detailed execution while reducing wall time, transient allocation, and retained memory?
5. Do atom selection and RHF coordinate blocking reduce coordinate-dependent work and memory without changing selected gradient rows?
6. Does the current validation-transaction design avoid repeated expensive DFT state audits during one public calculation while still protecting the scientific result?
7. Do scanner sequences remain correct and efficient across forward, backward, and revisited geometries?
8. Do force-data reading, batch-integrity verification, evaluation, training, and checkpoint restart remain practical when relaxed Jacobians become large?
9. How do the current implementation, current direct backend, current dense debug oracle, and the archived pre-matrix-free implementation compare in wall time and peak memory?
10. At what response dimension, condition range, atom count, and basis size do the current methods become limited by iterations, AO response actions, nuclear derivative construction, DFT quadrature, memory, or state validation?

## Current comparison targets

Every result must identify one of the following targets rather than using the generic label `reference`:

| ID | Target | Role |
| --- | --- | --- |
| `pyscf-native` | Native PySCF reference energy and analytic gradient, with full grid response enabled for RKS and UKS | Zero-correction energy and gradient oracle on the same energy surface |
| `fresh-fd` | Central finite differences of complete DeePHF energy with a fresh PySCF SCF calculation at every displacement | Independent nonzero-correction gradient oracle |
| `direct-compact` | Current direct backend with `retain_details=False` | Production direct-response comparator |
| `direct-detailed` | Current direct backend with `retain_details=True` | Detailed descriptor and response-partition oracle |
| `zvector-compact` | Current matrix-free GMRES Z-vector backend with `retain_details=False` | Production scalar-adjoint path |
| `zvector-detailed` | Current matrix-free GMRES Z-vector backend with `retain_details=True` | Detailed adjoint-partition comparator |
| `dense-replay` | Validation-only explicit response matrix reconstructed from the same physical action and solved with `numpy.linalg.solve(A.T, b)` | Bounded-size linear-solver oracle |
| `pre-matrix-free` | Revision `5ad3b08` in an isolated worktree and process | Historical dense-adjoint accuracy, time, and memory baseline |

The historical comparison must use the same geometry, basis, projector, model checkpoint, numerical tolerances, thread controls, and dependency versions whenever both revisions accept the input. A historical timeout or memory-limit exit must be recorded as a measured scalability boundary rather than converted into a numerical value.

## Reproducibility contract

All measurements must run from a clean, recorded Git revision through `uv run` in a locked Python 3.11 environment. The run manifest must include the Git revision, diff status, `uv.lock` hash, Python, NumPy, SciPy, PyTorch, PySCF, LibXC, BLAS vendor, CPU model, physical and logical cores, RAM, operating system, process affinity, thread environment, and validation configuration hash.

Use two execution profiles:

- `deterministic-1t`: one physical CPU core with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and PyTorch intra-op and inter-op thread counts set to one.
- `throughput-8t`: eight pinned physical CPU cores with every numerical library limited to eight threads and no competing process on those cores.

Scientific acceptance uses `deterministic-1t`. Performance conclusions must be reported separately for both profiles and must never mix samples from different profiles.

Each timed backend or revision must run in a fresh child process so that peak resident memory and allocator history are attributable to one case. Record both cold end-to-end time, including reference construction and validation, and warm gradient time on a preconverged unchanged reference. Use one untimed warm-up followed by five repetitions for short cases, three repetitions for extended cases, and one complete measurement for maximum-tier cases. Report raw samples, median, minimum, maximum, median absolute deviation, and exit status.

Peak memory must be collected per child process with `/usr/bin/time -v` or an equivalent operating-system counter. Python `tracemalloc` may be added for Python allocation attribution but cannot replace process peak RSS because NumPy, PySCF, BLAS, and Torch allocate outside the Python heap.

## Frozen scientific inputs

Use spherical Gaussian bases, disabled molecular symmetry, real canonical orbitals, double precision, and tightly converged native references. Use the current accepted pure-LDA functional and deterministic unpruned grid for RKS and UKS. Record the complete grid provenance for every DFT case.

Use one frozen deterministic nonlinear `tanh` CorrNet for inference comparisons. The checkpoint must have an exact hash and must satisfy the current force-capable model contract. Scale its output so that the correction remains finite and the response contribution is numerically visible without overwhelming the native energy scale.

Use the established projector basis unless a task explicitly studies projector scaling:

```yaml
projector_basis:
  - [0, [0.8, 1.0]]
  - [1, [0.3, 1.0]]
```

Every accepted nonzero-correction case must satisfy these anti-vacuity requirements:

```text
max(abs(correction_gradient_response)) >= 1.0e-4 Eh/Bohr
max(abs(dq_dR_response)) >= 1.0e-4 Bohr^-1
explicit-only gradient error >= 10 * complete analytic gradient error
```

Every displaced reference must preserve electron count, spin, occupation counts, AO labels, occupied and virtual dimensions, and the intended electronic state. Record the minimum occupied-virtual gap and occupied-subspace overlap between the central and displaced references. A state change invalidates the finite-difference sample.

## Molecular workload matrix

All geometries must be deterministic, asymmetric, displaced from equilibrium, and free of exact point-group symmetry. Planar molecules must receive a small out-of-plane displacement so that Cartesian components cannot pass through symmetry-enforced zeros.

| Case | System | Reference families | Basis | Intended scale and role |
| --- | --- | --- | --- | --- |
| `S1` | Distorted formaldehyde | RHF, RKS | `6-31G` | Small scientific anchor, complete finite differences, dense replay, and historical baseline |
| `S2` | Distorted formaldehyde | RHF, RKS | `def2-TZVP` | Basis-growth anchor, dense replay when within the configured validation limit, and compact/detail comparison |
| `S3` | Distorted hydroxymethyl radical | UHF, UKS | `def2-SVP` | Coupled alpha/beta response, open-shell state continuity, and bounded dense replay |
| `L1` | Asymmetric distorted water tetramer | RHF, RKS | `def2-SVP` and `def2-TZVP` | Noncovalent forces, coordinate growth, DFT grid cost, blocking, and full direct/Z-vector timing |
| `L2` | Distorted benzene | RHF, RKS | `def2-SVP` | Large occupied-virtual space, many nuclear right-hand sides, compact execution, and rotation/permutation checks |
| `L3` | Distorted phenoxy radical | UHF, UKS | `def2-SVP` | Large coupled spin response and expensive finite-grid matrix-free actions |
| `X1` | Asymmetric water hexamer | RHF | `def2-TZVP` | Maximum-tier response-dimension, coordinate-count, memory, blocking, and selected-atom stress test |

Before accepting a geometry into the frozen matrix, run the native stability and state-continuity preflight, inspect descriptor eigenvalue gaps, record exact dense operator diagnostics when the response dimension permits, and perform a short direct/Z-vector pilot. Replace an unstable geometry with another deterministic distortion instead of weakening the scientific controls.

## Task A: complete scientific accuracy matrix

For `S1`, run all Cartesian components at central-difference steps `1.0e-3`, `3.0e-4`, and `1.0e-4 Bohr`. Every displaced point must construct a fresh PySCF reference and a fresh DeePHF method.

For `S2`, `S3`, `L1`, `L2`, and `L3`, use two complementary finite-difference sets:

- Evaluate at least twelve fixed Cartesian components covering every element type, every molecular region, all three coordinate axes, intramolecular and intermolecular motion, and both spin-bearing and non-spin-bearing centers where applicable.
- Evaluate five deterministic normalized full-coordinate directions and compare the directional derivative `dot(gradient, direction)` with a two-sided complete-energy displacement along the same direction. Directional differences exercise every atom with two SCF calculations per direction and are preferable to silently reducing the test to a few local components.

Use steps `3.0e-4` and `1.0e-4 Bohr` for long cases and retain `1.0e-3 Bohr` when the smaller steps do not exhibit a stable plateau. Record every component and directional error, not only the maximum.

For every family and case, compare zero-correction energy and gradient with native PySCF, compare direct-detailed with fresh finite differences of the complete energy and relaxed descriptor, compare direct-compact with direct-detailed, compare Z-vector compact and detailed results, and compare direct with Z-vector for reference, explicit correction, response correction, complete correction, and total gradients wherever those partitions are available.

### Accuracy acceptance criteria

| Quantity | Acceptance criterion |
| --- | --- |
| Zero-correction energy versus native PySCF | Maximum absolute error no greater than `1.0e-12 Eh` |
| Zero-correction HF gradient versus native PySCF | Maximum absolute error no greater than `1.0e-10 Eh/Bohr` |
| Zero-correction DFT gradient versus native PySCF | Maximum absolute error no greater than `1.0e-9 Eh/Bohr` |
| Direct versus Z-vector total gradient | Maximum absolute error no greater than `1.0e-8 Eh/Bohr` |
| Compact versus detailed gradient for one backend | Maximum absolute error no greater than `1.0e-11 Eh/Bohr` |
| Analytic gradient versus complete-energy finite difference | Maximum absolute error no greater than `1.0e-5 Eh/Bohr` |
| Directional analytic derivative versus complete-energy finite difference | Maximum absolute error no greater than `1.0e-5 Eh/Bohr` |
| Relaxed descriptor Jacobian versus fresh-reference finite difference | Maximum absolute error no greater than `1.0e-5 Bohr^-1` |
| Matrix-free Z-vector versus dense replay solution on bounded cases | Relative L2 error no greater than `1.0e-9` and maximum absolute error no greater than `1.0e-10` |
| Matrix-free residual | Maximum residual no greater than the configured strict residual tolerance |
| Checkpoint reload and repeated unchanged input | Identical energy and force arrays |

For isolated HF molecules, require the maximum absolute component of the total force sum to be no greater than `1.0e-8 Eh/Bohr`. Also test rigid translation, rigid rotation, atom-record permutation, and Angstrom/Bohr input equivalence on `S1`, `L1`, and `L2`. DFT invariance reports must retain the finite-grid error separately from the analytic-force error.

## Task B: matrix-free solver versus explicit dense replay

For every case with response dimension at or below a configured dense limit, reconstruct the physical occupied-virtual response matrix in validation code by applying the action-only operator to basis-vector blocks. Do not call or modify a production dense solver path. Verify symmetry explicitly, compute exact eigenvalues and condition number, solve `A.T z = b` with `numpy.linalg.solve`, and compare the dense solution, physical residual, response-gradient contraction, and final nuclear gradient with the matrix-free GMRES result.

Measure the following phases in separate processes:

1. Action-only problem construction.
2. Explicit matrix construction.
3. Dense factorization and solve.
4. Matrix-free GMRES solve with the production orbital-gap preconditioner.
5. Post-solve AO density and nuclear contraction.

Record response dimension, occupied and virtual counts, exact spectral range, exact condition number, Krylov restart, iteration count, forward and transpose action counts, preconditioner action count, solver residual, wall time, and peak RSS.

The dense replay is an oracle and benchmark only. The production matrix-free run must be instrumented to fail if it invokes `_response_operator_matrix_and_diagnostics`, creates an array with shape `(response_dimension, response_dimension)`, or calls `numpy.linalg.solve` for the adjoint equation.

Add an operator-conditioning sweep on a bounded distorted H4 or bond-stretched closed-shell series. At each geometry, record the exact operator spectrum, orbital gap, GMRES iterations, residual, direct/Z-vector agreement, and timing. Only scientifically accepted states enter accuracy statistics; a strict rejection is recorded with its exact reason and is never converted into a relaxed-tolerance pass.

## Task C: end-to-end backend scaling

For every molecular case, benchmark `direct-compact`, `direct-detailed`, `zvector-compact`, and `zvector-detailed` on the same preconverged reference and model. Measure native gradient time separately so response-path comparisons are not obscured by a shared PySCF cost. Also measure complete public `method.gradient(...)` time because users experience the whole transaction rather than isolated solver kernels.

Report these ratios for each case and execution profile:

```text
zvector_compact_time / direct_compact_time
zvector_compact_peak_rss / direct_compact_peak_rss
zvector_compact_time / zvector_detailed_time
zvector_compact_peak_rss / zvector_detailed_peak_rss
current_zvector_time / pre_matrix_free_zvector_time
current_zvector_peak_rss / pre_matrix_free_zvector_peak_rss
```

Fit wall-time and incremental-memory trends against response dimension, AO count, atom count, and nuclear right-hand-side count. Report the fitted slopes with the raw data; do not infer matrix-free scaling from one molecule.

### Performance acceptance rules

1. Scientific acceptance must pass before any speedup is reported.
2. A timed sample set is valid only when `MAD / median <= 0.05`; otherwise rerun after identifying host noise.
3. `zvector-compact` must perform exactly one scalar adjoint solve, use GMRES, report a positive iteration count, and avoid dense response-matrix materialization.
4. Compact and detailed modes must be numerically equivalent; compact mode must not have a higher median time or peak RSS on `L1`, `L2`, `L3`, or `X1`.
5. On at least two large closed-shell cases, `zvector-compact` must be faster than `direct-compact`; otherwise the scalar-adjoint efficiency objective is not demonstrated.
6. For the largest case completed by both implementations, the current matrix-free path must use less peak RSS than `pre-matrix-free` and `dense-replay`.
7. Any current-versus-archived slowdown greater than 20 percent with comparable variability must be reported as a regression and attributed to a measured stage before acceptance.
8. An archived dense run that exceeds its resource limit does not satisfy the comparison; the report must retain its timeout, signal, or peak-memory termination and show that the current run completed under the declared limit.

The first completed campaign establishes machine-specific timing baselines. Subsequent runs on the same dedicated host use the stored medians and variability to gate regressions; shared CI workers use structural counts and scientific thresholds but do not enforce absolute wall-clock limits.

## Task D: compact execution and redundant-work budgets

The audit findings identify cases where small results were returned only after large intermediates had already been constructed. Long-running validation must therefore measure peak work rather than only inspect returned driver attributes.

For direct and Z-vector backends in all four reference families, compare `retain_details=False` and `retain_details=True` with validation-only counters around descriptor projection, descriptor differential construction, AO density transformation, induced-potential evaluation, DFT grid traversal, native gradient evaluation, response solve, adjoint solve, integrity hashing, and large-array allocation.

The compact path must satisfy these current algorithmic expectations:

- One model sensitivity evaluation per public gradient transaction.
- One contracted descriptor differential producing the explicit gradient and AO correction potential without constructing complete `dq/dP` or `dq/dR` tensors.
- One complete restricted AO density response transformation or one complete transformation per unrestricted spin for compact direct response.
- One compact adjoint response-gradient result without constructing the public detailed adjoint partition object.
- One native DFT gradient evaluation per compact RKS or UKS gradient.
- No retained response object, adjoint object, relaxed descriptor Jacobian, or additive partition arrays after compact return.

Counters that depend on private functions are validation telemetry, not public API requirements. They must be versioned with the measured revision and may be updated when an implementation is legitimately simplified, provided the scientific outputs and higher-level work budgets remain enforced.

## Task E: atom selection and coordinate blocking

Use `L1` and `X1` to compare full-atom calculations with deterministic subsets containing one atom, one complete water monomer, half the atoms, and all atoms in permuted order. Compare selected results with the corresponding rows of a separately computed full gradient for direct and Z-vector backends.

For RHF direct response, benchmark `coordinate_block_size` values `1`, `2`, `4`, `8`, and the full selected atom count. Record block count, maximum residual, wall time, peak RSS, and final gradient error relative to the unblocked calculation.

Selected calculations must never construct coordinate arrays whose leading atom dimension exceeds the selection. Z-vector atom selection does not reduce the occupied-virtual solve dimension, so its solver time and nuclear-contraction time must be reported separately rather than presenting the total ratio as a response-solve speedup.

Acceptance requires selected gradient rows to agree with full gradient rows within `1.0e-10 Eh/Bohr`, every block residual to satisfy the strict response tolerance, and smaller coordinate blocks to reduce peak coordinate-dependent memory on `X1`. Timing tradeoffs must be reported without assuming that the smallest block is fastest.

## Task F: DFT validation and grid-response cost

Use RKS `S2`, RKS `L1`, UKS `S3`, and UKS `L3` with the exact supported LDA and grid configuration. Measure cold first validation, warm validation of an unchanged reference, complete direct gradient, complete Z-vector gradient, native PySCF gradient, induced-potential actions, and nuclear grid-coordinate and grid-weight contractions.

Run a sequence of ten repeated gradients on one unchanged reference and a sequence of ten fresh but numerically identical references. The unchanged-reference sequence evaluates transaction and cache reuse; the fresh-reference sequence establishes the unavoidable validation cost. All gradients must remain identical.

Record full scientific-state fingerprint counts, complete grid traversals, grid builds, native-gradient calls, NumInt kernel calls, LibXC calls, wall time, and peak RSS. The warm unchanged-reference path must avoid repeated full audit work within one public transaction, while the fresh-reference path must continue to validate every new scientific state.

For every DFT finite-difference displacement, rebuild the deterministic grid through the accepted workflow. Compare complete analytic gradients with fresh-reference finite differences and report fixed-grid, grid-coordinate, grid-weight, and total response contributions when detailed mode is requested.

## Task G: scanner endurance and cache freshness

Use the supported RHF gradient scanner on a deterministic 100-frame water-cluster trajectory containing forward motion, reverse motion, repeated frames, and abrupt returns to the initial geometry. Include bond stretches, angular distortions, intermolecular translations, and rigid-body rotations.

At frames `0`, `1`, `9`, `10`, `49`, `50`, `98`, and `99`, compare scanner energy and gradient with a fresh PySCF and DeePHF construction. Revisited geometries must reproduce their previous energy and gradient, and adjacent frames must remain finite and continuous when the electronic state is unchanged.

Report first-frame, median subsequent-frame, repeated-frame, and fresh-reconstruction times; per-frame peak RSS in child-process blocks; SCF iterations; GMRES iterations; and response residuals. Track process RSS over the 100-frame sequence to detect monotonic cache growth.

## Task H: force-data, reader, training, and evaluation throughput

Generate deterministic RHF relaxed-force datasets for water clusters at increasing sizes and frame counts. Use at least three Jacobian payload scales, including one dataset whose `dq_dR_relaxed` storage exceeds one GiB, subject to the declared host resource limit.

Measure dataset generation time per frame, serialization time, load-time validation, reader construction, batch issuance, per-batch integrity verification, one force contraction, one evaluation epoch, one training epoch, checkpoint save, checkpoint reload, and saved-data testing. Record CPU time, wall time, peak RSS, bytes read, bytes hashed, and Jacobian bytes per frame.

Compare energy-only and energy-plus-force evaluation on the same frame count, and compare force batch sizes `1`, `4`, `16`, and the largest size that fits the resource envelope. Keep the model and optimizer state fixed when measuring batch-validation cost.

For correctness, use a deterministic teacher CorrNet for the automated workflow and PySCF RMP2 energy and analytic-gradient targets for the physical water-cluster workflow. Require checkpoint restart to reproduce pre-restart predictions, reader-issued content to remain tied to its frame provenance, and all reported energy and force metrics to be finite and separately aggregated.

The throughput report must identify hashing and host-device conversion time separately from model forward, autograd sensitivity, Jacobian contraction, optimizer, and metric aggregation time. A performance improvement is valid only when the same mutation-detection and provenance contract remains active.

## Task I: cross-revision comparison

Create isolated Git worktrees for revision `5ad3b08` and the revision under test. Do not share Python module imports between revisions; execute each case in a fresh process with explicit source paths and write results to revision-specific output directories.

Use `S1`, `S2`, `S3`, `L1`, and `L2` wherever the archived revision accepts the current scientific input. Compare energies, descriptors, gradients, forces, response dimensions, solver identity, wall time, and peak RSS. The current result must first agree with `direct-compact` and `fresh-fd`; agreement with the archived result is additional regression evidence and cannot override an independent oracle failure.

The comparison report must distinguish improvements from changed work. Record whether each revision performs dense matrix construction, exact operator diagonalization, repeated model sensitivity evaluation, repeated DFT grid audit, detailed Jacobian construction, response partition construction, batch hashing, and retained-result materialization.

## Resource tiers

The maximum wall-time allocation for any child process is three hours.

| Tier | Intended tasks | Wall-time limit per child process | Peak RSS limit |
| --- | --- | --- | --- |
| `bounded` | `S1`, dense replay, focused compact/detail checks | 30 minutes | 8 GiB |
| `extended` | `S2`, `S3`, `L1`, `L2`, DFT cache sequences, medium datasets | 90 minutes | 32 GiB |
| `maximum` | `L3`, `X1`, one-GiB Jacobian dataset, complete cross-revision scaling | 3 hours | 64 GiB |

Resource limits are part of the result. A timeout, out-of-memory exit, nonconvergence, or strict scientific rejection must preserve stdout, stderr, exit status, last diagnostics, and resource counters.

## Report schema

Each child process must write one JSON result containing:

- Case, reference family, geometry hash, charge, spin, basis, AO count, occupied and virtual counts, response dimension, atom count, selected atoms, coordinate block size, projector hash, model hash, grid hash, and numerical controls.
- Backend, result mode, solver, solve count, iteration count, restart, residual, orbital gap, exact spectral data when available, action counts, and scientific-state validation counts.
- Energy partitions, gradient partitions when requested, force, invariance residuals, finite-difference values, error norms, and the worst atom and coordinate.
- Cold wall time, warm wall-time samples, CPU time, peak RSS, Python peak allocation when collected, and stage-level timings.
- Revision, dependency and hardware metadata, process affinity, thread profile, raw stdout and stderr paths, exit status, timeout status, and pass/fail reasons separated into `scientific`, `integrity`, `performance`, and `resource` categories.

The campaign summary must not collapse all outcomes into one Boolean without preserving category failures. It must include comparison tables for accuracy, median time, peak RSS, GMRES iterations, direct/Z-vector ratios, compact/detailed ratios, current/archived ratios, and scaling fits.

## Execution order

1. Verify the locked environment and run `tests/response_scalability`, the focused direct-gradient objectives, the focused Z-vector objectives, and the baseline suite.
2. Freeze geometries, checkpoint, projector, finite-difference directions, thread profiles, and resource limits; record their hashes.
3. Run `S1` complete scientific acceptance and dense replay before any large timing campaign.
4. Run `S2` and `S3` to validate basis growth, spin coupling, dense replay, and compact/detail equivalence.
5. Run `L1` and `L2` for direct/Z-vector, coordinate-blocking, atom-selection, DFT, and scaling measurements.
6. Run scanner endurance and the medium force-data throughput matrix.
7. Run `L3`, `X1`, the largest force dataset, and the cross-revision campaign under the three-hour maximum resource tier.
8. Re-run every failed or noisy performance case once on the same dedicated host, preserving both result sets.
9. Generate the machine-readable aggregate and a concise Markdown report containing exact commands, scientific errors, timing ratios, peak-memory ratios, and unresolved limits.

## Intended artifact layout

```text
validation/scientific_performance/
  README.md
  geometries/
  configs/
  checkpoints/
  scripts/
  reports/
  runs/
```

Version only compact deterministic inputs, scripts, schemas, and accepted summary reports. Write large generated datasets, checkpoints used only by one run, raw arrays, process logs, and temporary worktrees to explicit ignored or external output directories. Executable regression tests distilled from confirmed defects belong under objective-specific directories in `tests/`.

## Completion criteria

The campaign is complete when every current reference family has passed independent scientific checks at both bounded and long scale, matrix-free and dense solutions agree on bounded chemical operators, direct and Z-vector gradients agree on large systems, finite differences validate nonzero response contributions, compact execution demonstrates lower work or memory than detailed execution, selected and blocked calculations reproduce full results, DFT transaction reuse is quantified, scanner and training endurance complete without state or memory drift, and timing and peak-memory comparisons are published for current direct, current matrix-free Z-vector, bounded dense replay, and the archived dense-adjoint baseline.
