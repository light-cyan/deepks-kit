# Current Project Audit

Date: 2026-08-24

Revision reviewed: `adbd2fe`

Scope: current production code, current non-legacy tests, packaging metadata, and the current working-tree integration state.

## Executive summary

The current implementation passes its default test suite, but it still contains correctness hazards, release-state defects, expensive validation in production hot paths, duplicate resident data, and substantial backend duplication.

The highest-risk scientific issue is that the matrix-free adjoint path treats `apply` as `apply_transpose` and uses one sampled bilinear identity as its symmetry check. The current telemetry can miss an unstable mode, and production accepts stability and condition controls that do not gate production execution.

The highest-cost execution issues are repeated descriptor finite differences, repeated response and adjoint audits, full DFT grid finite differences during reference validation, and per-sample SHA-256 validation inside every training batch.

The current Git integration state also requires attention: `pyproject.toml` declares `README.md` as package metadata, while the Git index tracks `README.old.md` and the current `README.md` is untracked.

## Severity definitions

- P0: Produces unavoidable corruption or unusable results in the supported path.
- P1: Can produce incorrect scientific behavior, blocks a reliable release, or causes dominant avoidable cost in a normal supported path.
- P2: Affects boundary correctness, scalability, memory consumption, or verification reliability.
- P3: Structural debt that materially increases future defect and maintenance risk.

## Findings

### F01 — P1 — Public explicit-derivative atom selection accepts invalid indices and can return incorrect values

Evidence: `deepks/deephf/method.py:312-317` forwards `atom_indices` without validation, and `deepks/descriptor/derivatives.py:19-26` converts the input to a tuple and builds a dictionary that silently collapses duplicate indices.

Evidence: `deepks/descriptor/derivatives.py:45-60` indexes `aoslice_by_atom()` directly and adds projector motion only to the last dictionary position associated with a duplicated raw atom.

Reproduction on a two-atom RHF molecule: `dq_dR_explicit(atom_indices=[-1])` returned shape `(1, 3, 2, 1)` instead of rejecting the negative index; `[0, 0]` returned shape `(2, 3, 2, 1)` with projector motion assigned only to the second duplicate; `[2]` failed later with a generic NumPy `IndexError`.

Impact: callers of the public derivative API can receive a finite tensor with the requested shape but incorrect nuclear derivatives.

Required invariant: raw atom indices must be validated centrally as unique, non-boolean integers in `[0, natm)`, with one documented empty-selection behavior shared by all gradient and derivative entry points.

### F02 — P1 — Matrix-free transpose correctness depends on an unproved symmetry assertion

Evidence: RHF, RKS, and UHF implement `apply_transpose` by returning `apply` in `deepks/deephf/pyscf_rhf.py:2424-2426`, `deepks/deephf/pyscf_rks.py:1358-1365`, and `deepks/deephf/pyscf_uhf.py:2360-2362`.

Evidence: `deepks/deephf/adjoint.py:285-296` checks symmetry with one fixed pair of vectors and one scalar bilinear identity.

Impact: an implementation defect or numerical asymmetry that is orthogonal to the fixed probe pair can pass the check; the GMRES solve, transpose residual, and physical residual then all reuse the same forward action and cannot independently detect the wrong transpose equation.

Required invariant: a production adjoint must either provide a mathematically independent transpose action or make operator symmetry an explicit trusted backend invariant supported by exact small-system regression checks and residual checks that do not claim independence from the same action.

### F03 — P1 — Production operator stability controls are accepted but do not control production execution

Evidence: `operator_stability_tolerance`, `operator_condition_tolerance`, and `operator_dimension_limit` remain public direct and Z-vector options in `deepks/deephf/method.py:42-67` and the spin/DFT method option sets.

Evidence: production diagnostics use the fixed 16-step telemetry in `deepks/deephf/adjoint.py:260-343`; exact stability and condition enforcement exists only in explicit `validate_response_operator_exact` methods such as `deepks/deephf/pyscf_rhf.py:1730-1739`.

Evidence: `tests/response_scalability/test_rhf_matrix_free_response.py:118-127` demonstrates a diagonal operator with a negative eigenvalue for which the production telemetry reports a positive minimum Ritz value.

Evidence: `tests/response_scalability/test_rhf_matrix_free_response.py:130-147` confirms that a production adjoint succeeds with `operator_dimension_limit=1` while its response dimension is four.

Impact: callers can configure named stability and condition controls and receive a production result even though those controls have not certified or rejected the operator.

Required invariant: production option names and diagnostics must describe controls that are actually enforced; bounded telemetry must remain observational and must not be exposed as certified eigenvalues or a certified condition number.

### F04 — P1 — Descriptor sensitivity performs finite-difference and state-integrity audits on every force evaluation

Evidence: `deepks/deephf/method.py:339-394` evaluates the differentiable result twice and compares it with a central finite difference over every descriptor scalar.

Evidence: `deepks/deephf/method.py:489-528` evaluates each ordinary energy four times for every displaced descriptor, and `deepks/deephf/method.py:530-562` performs positive and negative displacements for every scalar.

Cost: one `correction_sensitivity` call performs `6 + 8N` model outputs for `N` descriptor scalars. A three-atom, four-feature descriptor therefore performs 102 model outputs instead of one differentiable output.

Evidence: `deepks/deephf/capabilities.py:372-434` fingerprints parameters and buffers and then fingerprints the same tensor state again through `state_dict`; `deepks/deephf/capabilities.py:437-463` snapshots all initialized RNG states around ordinary model evaluations.

Impact: model cost scales with descriptor size even before response theory begins, and CUDA execution adds repeated device-to-host synchronization for state and RNG fingerprints.

Required invariant: production force evaluation must use one autograd sensitivity for one immutable model state; deterministic finite differences and exhaustive state audits must be explicit validation operations rather than per-force operations.

### F05 — P1 — Direct gradient backends recompute the same sensitivity and trusted response audits

Evidence: RHF direct calls `validate_force_compatibility` and later `correction_sensitivity` in `deepks/deephf/gradient.py:219-235`, causing two complete sensitivity evaluations.

Evidence: RKS direct calls compatibility validation, `response`, `dq_dR_response`, and final sensitivity in `deepks/deephf/rks_gradient.py:141-150`; `response` calls compatibility validation in `deepks/deephf/rks_method.py:102-114`, and `dq_dR_response` calls it again in `deepks/deephf/rks_method.py:163-173`.

Evidence: UHF and UKS follow the same four-sensitivity structure in `deepks/deephf/uhf_gradient.py:82-99` and `deepks/deephf/uks_gradient.py:51-64` through their method-level response consumers.

Cost: for a 12-scalar descriptor, RHF direct performs 204 model outputs and the RKS/UHF/UKS direct structure performs 408 model outputs before counting reference and response audits.

Impact: restricted and unrestricted backends have materially different execution efficiency for the same public operation, and DFT backends retain the most expensive legacy call composition.

Required invariant: one gradient call must compute one sensitivity and pass it, one trusted response, and one validated model/reference snapshot through the entire contraction path.

### F06 — P1 — Trusted response and adjoint results are rebuilt instead of consumed

Evidence: RKS stores the exact response and producing adapter in `deepks/deephf/rks_method.py:113-123`, but `_validate_response` still calls `adapter.audit_response_equations(response)` at `deepks/deephf/rks_method.py:126-149`.

Evidence: UHF constructs a new response adapter and performs a full equation audit for every response consumption in `deepks/deephf/uhf_method.py:187-201`.

Evidence: all Z-vector method paths call `audit_adjoint` on an adjoint just produced by the trusted adapter, including `deepks/deephf/method.py:839-862`, `deepks/deephf/rks_method.py:341-357`, `deepks/deephf/uhf_method.py:439-453`, and `deepks/deephf/uks_method.py:207-213`.

Impact: operator telemetry, fingerprints, induced potentials, nuclear derivative tensors, and gradient partitions are recomputed after a successful solve, often through a fresh adapter with no reusable cache.

Required invariant: an immutable result produced and sealed by the current method state must use constant-cost integrity checks at consumption; full equation reconstruction must be a separate audit operation.

### F07 — P1 — DFT reference validation rebuilds grids and performs nuclear finite differences on normal calls

Evidence: `deepks/deephf/pyscf_rks.py:831-857` builds a fresh central grid and requests a finite-difference audit of grid-weight derivatives.

Evidence: `deepks/deephf/pyscf_rks.py:581-620` rebuilds a strict grid at both displacement signs for every atom and Cartesian coordinate, producing `1 + 6 * natm` grid builds per provenance validation.

Evidence: `deepks/deephf/pyscf_rks.py:1045-1106` invokes this provenance path and then repeats native and independent Coulomb/XC quadrature construction in every `validate_rks_reference` call; UKS validation wraps the same finite-grid machinery.

Impact: validation cost can dominate RKS and UKS response and gradient execution and is multiplied by the repeated trusted-result audits in F06.

Required invariant: immutable DFT reference provenance and expensive scientific cross-checks must be established once per reference state; normal operator actions and result consumption must use the sealed state.

### F08 — P1 — Force training hashes and copies the complete Jacobian inside every batch

Evidence: `deepks/model/train.py:480-508` searches every configured contract and frame for each sample, copies energy, descriptor, force, and the five-dimensional relaxed Jacobian to CPU NumPy arrays, and validates each field.

Evidence: `deepks/data/force_schema.py:1519-1556` reparses the sealed contract, rebuilds the sample lookup, checks finiteness, and computes SHA-256 for every array.

Evidence: the same arrays and hashes have already been strictly validated by `load_force_dataset` at `deepks/data/force_schema.py:1627-1679` before Reader construction.

Measured impact: on the existing six-frame training sample with an approximately 20 KiB Jacobian, 20 evaluations took approximately 0.430 seconds with runtime sample validation and 0.047 seconds when only that redundant validator was bypassed, a 9.1-fold difference.

Required invariant: persisted identities must be validated at load time and associated with a precomputed sample-ID lookup; training batches must not copy GPU tensors to CPU or hash full Jacobians.

### F09 — P1 — Package metadata and the Git index disagree about the project README

Evidence: `pyproject.toml:9` declares `readme = "README.md"`.

Evidence: the current Git index tracks `README.old.md` and `README.draft.md`, while the current `README.md` is untracked and `README.draft.md` is deleted in the working tree.

Impact: the package metadata depends on a file that is outside the reviewed revision, so release and clean-checkout behavior depend on uncommitted local state.

Required invariant: every file referenced by package metadata must be tracked in the same revision that declares it.

### F10 — P2 — Matrix-free Krylov actions perform multiple full-vector copies and SHA-256 hashes

Evidence: `_isolated_problem_action` in `deepks/deephf/adjoint.py:203-257` copies the input into an immutable buffer, hashes it before the action, hashes it after the action, copies the result, and hashes the input again.

Evidence: the GMRES action and preconditioner then copy the validated output again at `deepks/deephf/adjoint.py:470-500`.

Measured behavior: `symmetric_operator_telemetry` applies a diagonal test operator 18 times; a one-iteration 128-dimensional GMRES solve still invoked seven transpose actions, one forward action, and three preconditioner actions.

Impact: the action-only solver avoids a dense matrix but adds allocation and hashing proportional to response dimension at every Krylov iteration.

Required invariant: trusted internal response problems must use a low-overhead checked action; mutation-fuzzing and cryptographic input checks must remain outside the iterative inner loop.

### F11 — P2 — Atom selection reduces work only for RHF direct gradients

Evidence: RHF direct propagates selected atoms into native gradients, explicit derivatives, Hamiltonian derivatives, and response blocks in `deepks/deephf/gradient.py:209-275`.

Evidence: RKS and UHF direct compute complete results in `_kernel` and slice only after completion in `deepks/deephf/rks_gradient.py:235-248` and `deepks/deephf/uhf_gradient.py:201-214`; UKS inherits the UHF selection behavior.

Evidence: RHF Z-vector computes full native gradients, explicit derivatives, and nuclear contractions before slicing at `deepks/deephf/zvector.py:77-190`, with equivalent behavior in the spin and DFT Z-vector drivers.

Boundary inconsistency: `kernel(atmlst=[])` raises `ValueError: response atom_indices must not be empty` in RHF direct, while slice-based backends can return an empty selection after computing all atoms.

Impact: local-force and selected-atom calls have backend-dependent behavior and usually retain full-system nuclear cost.

Required invariant: all backends must share the same atom-selection contract and pass selected atoms to every coordinate-dependent contraction that can support a subset.

### F12 — P2 — Force Reader retains duplicate full datasets and Jacobians

Evidence: `deepks/model/reader.py:68` retains every loaded canonical NumPy array in `_force_arrays`.

Evidence: `deepks/model/reader.py:188-236` retains NumPy references in `data_energy` and `data_descriptor` and creates copied Torch storage for energy, descriptor, force, and the complete relaxed Jacobian.

Impact: the largest five-dimensional tensor is resident at least twice before any device transfer, while canonical base/target fields that training does not consume also remain resident.

Required invariant: Reader must retain one authoritative storage representation per field and load or map large Jacobians in bounded batches when the dataset exceeds memory.

### F13 — P2 — Evaluation materializes all results and preserves unnecessary autograd graphs

Evidence: `evaluate_reader` creates a list containing every `EvaluationResult` in `deepks/model/train.py:272-278` even though summary metrics are already detached scalar aggregates.

Evidence: energy-only prediction executes the model with normal gradient tracking in `deepks/model/evaluate.py:113-131`; the returned energy, loss, and prediction remain reachable through the result list.

Impact: validation memory grows with the number of batches, and energy-only evaluation pays autograd construction cost without using gradients.

Required invariant: evaluation must aggregate detached counts and error sums online, use `torch.no_grad()` for energy-only metrics, and limit gradient-enabled scope to force differentiation.

### F14 — P2 — Training control values are not validated before arithmetic and scheduling

Evidence: `deepks/model/train.py:669-674` constructs Adam and StepLR directly from public inputs.

Evidence: when `stop_lr` is set, `deepks/model/train.py:670-672` divides by `n_epoch // decay_steps`; any positive `n_epoch < decay_steps` produces division by zero.

Evidence: `display_epoch` is used as a modulo divisor at `deepks/model/train.py:728` without a positive-integer check.

Impact: valid-looking short training configurations can fail before training or during the first display decision with low-level arithmetic errors.

Required invariant: epoch counts, display intervals, decay intervals, learning rates, and factors must be type-, range-, and cross-field-validated before model or optimizer mutation.

### F15 — P2 — Force-schema operator diagnostics have mixed exact and estimated semantics

Evidence: `operator_diagnostics_are_estimates` is optional during frame normalization in `deepks/data/force_schema.py:915-929`.

Evidence: stability and condition tolerance enforcement is absent from the normalized frame checks at `deepks/data/force_schema.py:978-1049`; only positivity, ordering, and a formula based on the minimum Ritz endpoint remain.

Evidence: `deepks/data/force_schema.py:1035-1045` computes the condition estimate denominator from `abs(minimum)` rather than the smallest absolute spectral value; for an indefinite estimate this can report one even when endpoint magnitudes differ substantially.

Impact: manifests with and without the estimate flag are accepted under one schema while the numeric fields no longer carry one consistent scientific meaning.

Required invariant: persisted fields must distinguish sampled Ritz telemetry from exact spectral diagnostics, and validation rules must be selected by that explicit semantic type.

### F16 — P2 — Test collection depends on directory order

Evidence: several test modules import shared values with top-level `from conftest import ...`, including `tests/rks_zvector_inference/test_rks_adjoint_strict.py:15`, `tests/rks_analytic_forces/test_rks_gradient.py:7`, and force-training tests.

Reproduction: one combined targeted command that listed UHF Z-vector tests before RKS Z-vector tests failed collection because the RKS test imported `_P4B_FIXTURES` from `tests/uhf_zvector_inference/conftest.py`.

Impact: the default suite can pass while legitimate targeted combinations fail or import fixtures from the wrong objective directory.

Required invariant: reusable helpers must live in importable uniquely named support modules, while `conftest.py` must be consumed only through pytest fixture discovery.

### F17 — P2 — Public dependency metadata does not match the runtime PySCF contract

Evidence: `pyproject.toml:25-32` declares an unconstrained `pyscf` dependency.

Evidence: `deepks/deephf/pyscf_rhf.py:491-498` and corresponding backend validators reject every PySCF series except 2.14.

Impact: dependency resolution and installation can succeed with a version that production code immediately rejects.

Required invariant: install-time dependency constraints and runtime capability constraints must describe the same supported version range.

### F18 — P3 — Backend monoliths duplicate validation, fingerprint, response, and result representations

Evidence: current production Python totals approximately 29,155 lines; `deepks/deephf/pyscf_rks.py`, `pyscf_uhf.py`, and `pyscf_rhf.py` contain 4,154, 3,281, and 3,272 lines respectively.

Evidence: `_immutable_array`, `_update_fingerprint_value`, `_validated_float64_array`, `_cycle_limit`, version parsing, reference validation, integrity hashing, operator telemetry, and result auditing are implemented repeatedly across backend modules.

Evidence: response dataclasses retain total, occupied-virtual, metric, coefficient, and density representations even when totals are reconstructible from their components.

Impact: fixes such as RHF selected-atom propagation and trusted-result handling do not automatically reach RKS, UHF, and UKS, producing the current behavioral divergence and repeated computation.

Required invariant: shared validation and linear-response mechanics must have one implementation, and result objects must store a minimal canonical representation with derived views computed only when requested.

### F19 — P3 — Performance verification records timings without enforcing regression limits

Evidence: `validation/scripts/run_performance.py:1` explicitly states that it runs without timing gates.

Evidence: the `passed` result beginning at `validation/scripts/run_performance.py:187` checks scientific diagnostics rather than timing, action-count, allocation, or peak-memory budgets.

Impact: the default 853-test suite can remain green while a normal gradient or training batch acquires hundreds of additional model evaluations, grid builds, hashes, or tensor copies.

Required invariant: performance-critical paths need deterministic structural budgets, including model-output counts, operator-action counts, prohibited device-to-host transfers, bounded retained tensors, and separate coarse timing ceilings.

## Verification evidence

- `uv run pytest -q`: 853 passed in 523.92 seconds.
- Combined targeted suite with reordered objective directories: failed collection through a `conftest` module collision as described in F16.
- Matrix-free action-count experiment: telemetry used 18 forward actions; a one-iteration diagonal GMRES solve used seven transpose actions, one forward action, and three preconditioner actions.
- RHF atom-selection experiment: empty gradient selection raised a response error; negative and duplicate explicit-derivative indices were accepted; a positive out-of-range index failed only through downstream array indexing.
- Working-tree integration check: `README.md`, `docs/ResponseScalabilityReview.md`, and `draft.md` are untracked; `README.draft.md` is deleted; this audit does not modify those paths.

## Priority order

1. Restore scientific contract clarity for transpose actions, stability controls, and spectral telemetry in F02, F03, and F15.
2. Correct public atom-selection behavior and release metadata in F01, F09, F11, and F17.
3. Remove repeated sensitivity, trusted-result, DFT provenance, and force-sample audits from production hot paths in F04 through F08.
4. Remove Krylov-loop hashing, duplicate dataset storage, and retained evaluation graphs in F10, F12, and F13.
5. Repair test isolation and establish structural performance gates before consolidating backend infrastructure in F16, F18, and F19.
