# DeePHF Change Requirements 2

## Purpose

This document defines the corrective work required after the evaluation-context and PySCF-adapter refactor. The work protects scientific results from mutable cache aliases, makes calculation transactions fail atomically when scientific state changes, completes function-level decomposition, and removes the remaining mechanical redundancy without giving up the achieved calculation budgets.

## Scope

The requirements cover `deepks/deephf/`, `deepks/descriptor/workspace.py`, the public RHF, UHF, RKS, and UKS DeePHF methods, direct and scalar-adjoint gradients, compact inference, detailed RHF force-data generation, architecture tests, and deterministic performance tests.

The current supported calculation surface remains RHF, UHF, RKS, and UKS references; direct and scalar-adjoint analytic gradients; atom selection; compact gradients and forces; detailed response data; scanners where implemented; deterministic DFT grids; and RHF force-data production.

## Current Acceptance State

The calculation refactor provides one `EvaluationContext` per public workflow, shared descriptor and model evaluation, a shared descriptor derivative workspace, atomic publication inside individual energy calls, compact zero-sensitivity bypasses, canonical array and fingerprint utilities, small public PySCF facades, isolated dense audit modules, independent direct and scalar-adjoint solvers, and deterministic operation counters.

The locked Python 3.11 suite currently contains 812 passing tests. The architecture and deterministic performance objective contains 20 passing tests. The RHF public workflow performs one AO-density construction, one descriptor evaluation, one model forward, and the backend-specific response work required by the selected compact or detailed path.

Acceptance remains incomplete because public numerical accessors can expose writable aliases of calculation-scoped caches, active transactions trust their entry fingerprint until exit, an exit-time state failure can leave published energy fields populated, the function-size check does not descend into audit modules, several audit functions remain above the required hard limit, and small exact code duplicates remain.

## Required Changes

### R1: Isolate Public Numerical Results from Evaluation Caches

Priority: Critical.

`EvaluationContext` and `DescriptorDerivativeWorkspace` must retain exclusive ownership of every cached numerical value. A public result must not share writable storage with a cached NumPy array or cached Torch tensor.

The rule applies to AO density, spin density, projected density, descriptor values, descriptor sensitivity, `dq/dP`, and any later accessor that exposes a calculation-scoped cached value. Internal consumers may share immutable cache storage when they do not expose it beyond the implementation boundary.

Public accessors may return an independent owned array or a read-only representation whose writeability cannot be used to modify the internal cache. A read-only flag on a NumPy view of writable Torch storage is insufficient when the caller can re-enable writeability or reach the writable base object.

The correction energy, descriptor sensitivity, explicit derivative, AO correction potential, direct response, adjoint response, gradient, and exported descriptor produced in one transaction must remain derived from the same unmodified descriptor state.

Acceptance criteria:

- `np.shares_memory` or an equivalent ownership check confirms that public NumPy results do not alias mutable internal caches.
- Attempting to modify the result of `ao_density`, `projected_density`, `descriptor`, `correction_sensitivity`, or `dq_dP` either raises immediately or changes only the caller-owned result.
- Modifying one public result inside `with method.calculation():` does not change a later energy, sensitivity, direct gradient, scalar-adjoint gradient, descriptor, or derivative result.
- A sensitivity returned to a caller cannot be changed to trigger the zero-sensitivity fast path in a later gradient calculation.
- Cache isolation does not add descriptor evaluations, model forwards, derivative-overlap integral evaluations, projected-density constructions, or shell-Jacobian constructions.
- Tests cover RHF and at least one unrestricted or DFT family, with both direct and scalar-adjoint consumers represented.

### R2: Detect Scientific-State Changes Before Cached Reuse

Priority: Critical.

An active `calculation()` transaction must not treat its entry fingerprint as proof that the model, reference, molecule, descriptor, device, or dtype remains unchanged. Before a cached value is reused across public calculation boundaries, the implementation must verify that the state on which the value depends is still current.

The guard should use low-cost version evidence where the runtime provides reliable mutation versions, including Torch tensor version counters, object identity, configuration identity, device, dtype, and explicit context generations. State without reliable version evidence must use a conservative fingerprint at the boundary where stale cache reuse could otherwise occur.

Correctness takes precedence over the two-boundary fingerprint budget. The implementation should distinguish complete scientific fingerprints from cheap state-version checks in its operation counters so that safety checks remain visible without classifying every version comparison as a full hash.

Acceptance criteria:

- Mutating a model parameter or buffer after one successful call inside `calculation()` causes the next cache-dependent public call to fail before returning a stale value.
- Changing reference orbitals, occupations, total energy, molecule state, descriptor state, model mode, device, or dtype inside a transaction cannot reuse an earlier dependent cache entry.
- Replacing the reference, molecule, descriptor, projector molecule, or model remains fail-closed.
- An unchanged `evaluate_molecule` workflow retains one descriptor evaluation and one model forward.
- Nested internal calls consume the accepted state guard without repeating equivalent checks at adjacent lines.
- Operation counters separately report full scientific fingerprints and cheap state-version validations.

### R3: Make the Public Calculation Transaction Atomically Fail

Priority: Critical.

`DeePHF.calculation()` must provide the same failure cleanup guarantee as an individually decorated energy or gradient call. A validation failure raised during outer transaction exit must clear every result published from that transaction.

The transaction must track participating method and gradient-driver result publishers. Cleanup must cover `e_base`, `e_corr`, `e_tot`, compact and detailed gradient fields, response or adjoint results, descriptor diagnostics, response diagnostics, and force-facing result aliases.

Local workflow dictionaries and persisted force-data records must be constructed only from a transaction that has completed its exit validation successfully.

Acceptance criteria:

- A model or reference mutation after a successful nested `kernel()` causes the outer transaction to fail with `e_base`, `e_corr`, and `e_tot` cleared.
- A gradient published before an exit-time failure is cleared from its driver, including detailed response or adjoint fields.
- `evaluate_molecule` returns no result after an exit-time state failure.
- RHF force-data generation writes no partial record after an exit-time state failure.
- A successful transaction continues to publish energy and gradient fields together with their existing shapes, dtypes, atom ordering, and finite-value guarantees.

### R4: Complete Function-Level Decomposition

Priority: High.

The source-size contract must cover every Python module under `deepks/deephf/`, including `deepks/deephf/audits/`. Moving a dense function into an audit package does not satisfy function-level decomposition.

Calculation, response, adjoint, reference-validation, and scientific-audit functions should remain at or below 100 physical lines and must remain below 200 physical lines. Small declarative assembly functions and dataclass definitions remain outside this calculation-function concern while still complying with module-size limits.

Large audit routines must be divided along scientific responsibilities such as input validation, independent invariant groups, residual reconstruction, partition reconciliation, reference-oracle comparison, and diagnostic assembly. Each helper must have a narrow data contract and must not solve a response or adjoint problem during an audit.

Acceptance criteria:

- The source constraint test uses recursive discovery and reports module-relative paths for every oversized function.
- No function under `deepks/deephf/` exceeds 200 physical lines.
- `audit_response_equations`, `audit_adjoint`, and reference-audit entry points remain orchestration functions with independently testable helpers.
- Dense audit modules remain isolated from compact production paths and are imported lazily by audit entry points.
- Direct and scalar-adjoint production solvers remain independent after decomposition.
- The complete scientific suite passes with unchanged numerical tolerances.

### R5: Define Model Validation Timing as a Public Contract

Priority: High.

Model validation must distinguish construction-time structural validation from calculation-time output validation. The chosen timing must be explicit and consistent across RHF, UHF, RKS, and UKS methods.

The recommended contract keeps construction free of a speculative model forward: construction validates model type, architecture, projector metadata, feature dimensions, real `torch.float64` parameters and buffers, finite state, evaluation mode, and supported device. The first calculation validates that the actual output is one finite real scalar and that its descriptor sensitivity satisfies the force contract.

This lazy-output contract preserves the one-forward workflow budget. Public API documentation and tests must identify output-shape and non-finite-output failures as calculation-time failures rather than presenting the changed timing as unchanged constructor behavior.

Acceptance criteria:

- Static model incompatibilities fail during method construction for every reference family.
- Output rank, output cardinality, output dtype, complex output, non-finite output, detached sensitivity, and unsupported differentiation fail during the first relevant calculation.
- A failed first calculation leaves method and driver result fields empty.
- The model is not evaluated solely for constructor validation.
- Tests use names and assertions that encode the selected public failure boundary.

### R6: Remove Remaining Mechanical Redundancy

Priority: Medium.

Consecutive identical state validations must be reduced to one boundary check. Exact response helper bodies shared by restricted Hartree-Fock and restricted Kohn-Sham implementations must have one method-neutral owner when their algebra and contracts are identical.

Unused cached-state properties must either become the canonical implementation used by all zero-sensitivity branches or be removed. Family constructors may retain explicit family policy, while identical argument normalization and base initialization should have one shared owner.

Acceptance criteria:

- Compact RKS, UHF, and UKS gradient assembly contains one state validation per semantic boundary.
- The identical restricted `_induced_mo_potential` contraction has one canonical implementation.
- Every exported or internal cache convenience property has at least one production consumer.
- Consolidation does not introduce imports from method-neutral algebra into PySCF, workflow, persistence, or model layers.
- Direct and scalar-adjoint code paths do not call one another after consolidation.

### R7: Expand Regression Budgets Across the Supported Families

Priority: High.

Deterministic budget tests must protect cache isolation and transaction safety in addition to repeated-work elimination. RHF retains detailed force-data coverage, while compact zero-sensitivity and nonzero-sensitivity coverage must exercise RHF, UHF, RKS, and UKS across both direct and scalar-adjoint backends.

The operation-count contract must define whether construction is inside or outside the reported scope. A workflow report that shows two transaction fingerprints must also state that initial method binding performs the third complete fingerprint.

Acceptance criteria:

- One unchanged public workflow performs one descriptor evaluation and one model forward for every supported reference family.
- Compact zero-sensitivity direct and scalar-adjoint gradients perform no derivative construction and no response or adjoint solve for every supported reference family.
- Detailed RHF force-data generation continues to materialize the response and relaxed descriptor derivatives for `model=None`.
- Mutation-isolation and mid-transaction state-change tests fail on the current unsafe behavior and pass after correction.
- Full fingerprints, cheap state guards, cache hits, cache invalidations, direct solves, adjoint solves, operator actions, and derivative materializations have separate counters.
- Scientific finite-difference and direct-versus-adjoint assertions remain independent from performance-budget assertions.

### R8: Keep Current Documentation and Repository State Coherent

Priority: Medium.

The current DeePHF requirements document must describe the current implementation and active acceptance gaps. Project audit records remain under `docs/legacy/project-audits/`, and current validation documentation must use those paths when it cites the archived audits.

Requirement documents referenced by implementation or review work must be tracked by Git. A completed implementation report must use the same counting scope and size scope as its tests, including constructor fingerprints and recursively discovered audit functions.

Acceptance criteria:

- `docs/DeePHFChangeRequirements2.md` is tracked with the repository changes that establish this acceptance baseline.
- References to project audit documents resolve under `docs/legacy/project-audits/`.
- Current documentation does not label pre-refactor line counts or operation counts as the current implementation state.
- Completion reports distinguish passing numerical regression tests from untested ownership and transaction guarantees.

## Implementation Order

1. Add failing cache-alias and mid-transaction mutation tests without changing production behavior.
2. Isolate public numerical results from internal cache storage.
3. Add cache-dependency state guards and atomic outer-transaction cleanup.
4. Re-run compact and detailed operation budgets and adjust only the explicitly defined state-guard counters.
5. Recursively enforce the function-size contract and decompose oversized audit functions.
6. Consolidate exact duplicates and remove unused cache helpers.
7. Align model-validation tests and current documentation with the selected public failure boundaries.
8. Run the objective-specific tests followed by the complete locked Python 3.11 suite.

## Verification

Run the focused objectives first:

```bash
uv run pytest tests/deephf_performance
uv run pytest tests/architecture
uv run pytest tests/analytic_forces tests/zvector_inference
uv run pytest tests/rks_analytic_forces tests/rks_zvector_inference
uv run pytest tests/uhf_analytic_forces tests/uhf_zvector_inference
uv run pytest tests/uks_analytic_forces tests/uks_zvector_inference
```

Run the complete locked environment verification afterward:

```bash
uv sync --locked --python 3.11
uv run pytest
```

For changes that affect packaging or public module exports, also run:

```bash
uv build
```
