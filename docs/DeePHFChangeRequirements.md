# DeePHF Change Requirements

## Purpose

This document defines the recommended implementation changes for the current perturbative DeePHF calculation path. The changes preserve the supported scientific behavior while reducing repeated computation, making result publication consistent, and decomposing oversized program files into cohesive modules.

## Scope

The requirements cover `deepks/deephf/`, the descriptor derivative facilities used by DeePHF, RHF force-data generation, and the objective-specific tests that protect these paths.

The supported calculation surface to preserve consists of RHF, UHF, RKS, and UKS references; direct and scalar-adjoint analytic gradients; compact inference; detailed response data used by force-data generation; atom selection; scanner execution; strict reference validation; deterministic DFT grid handling; and force-aware RHF training data.

## Current Assessment

The current scientific behavior is coherent within the tested support tier. Direct and Z-vector gradients are independently implemented, all four reference families have analytic-force coverage, bounded exact response audits remain separate from production matrix-free solves, and the complete locked Python 3.11 test suite passes.

The current implementation does not meet the desired efficiency and internal-cohesion standard. The main defects are repeated end-to-end evaluation, unnecessary response work for zero corrections, repeated derivative construction, full-buffer copies during integrity checks, duplicated family drivers, and compatibility adapters that combine several unrelated responsibilities.

### Measured Runtime Redundancy

One instrumented RHF sequence equivalent to `evaluate_molecule` produced the following counts after reference construction:

| Operation | Current count |
| --- | ---: |
| Scientific-state fingerprint calculation, including method construction | 13 |
| Descriptor evaluation | 4 |
| Correction-model forward evaluation | 3 |
| Direct response solve for `model=None` | 1 |

These counts are functional-regression targets. A passing numerical result does not justify retaining the repeated work.

### Oversized Program Files

| Program file | Current physical lines | Combined responsibilities |
| --- | ---: | --- |
| `deepks/deephf/pyscf_rks.py` | 3,958 | Reference contract, implementation identity, DFT functional and grid provenance, grid derivatives, direct response, adjoint response, dense audit, integrity, and native gradient |
| `deepks/deephf/pyscf_uhf.py` | 3,187 | Reference contract, unrestricted response, adjoint response, dense audit, integrity, and native gradient support |
| `deepks/deephf/pyscf_rhf.py` | 3,113 | Reference contract, scanner reference support, direct response, adjoint response, dense audit, integrity, and native gradient support |
| `deepks/deephf/pyscf_uks.py` | 1,187 | UKS reference validation, unrestricted DFT response composition, adjoint composition, integrity, and native gradient |
| `tests/analytic_forces/test_package_architecture.py` | 1,064 | Dependency rules, source ownership rules, import isolation, public-symbol ownership, and implementation-placement checks |

The largest response and adjoint audit functions are also several hundred lines long. File decomposition must therefore include function-level decomposition rather than moving unchanged large functions between modules.

## Required Changes

### R1: One Evaluation Context per Public Calculation

Priority: High.

The public molecular evaluation path must compute the AO density, descriptor, correction energy, and descriptor sensitivity through one scoped evaluation context. The same model forward must supply both correction energy and the autograd sensitivity needed by the gradient.

The context must be valid only for one bound reference, descriptor definition, model state, device, dtype, and calculation transaction. A state change must invalidate the context before any cached numerical value is consumed.

`evaluate_molecule` must execute energy, gradient, and descriptor publication under one outer scientific-state transaction. Nested calls must reuse the accepted state token and the evaluation context.

Acceptance criteria:

- One complete `evaluate_molecule` call performs no more than one descriptor evaluation and one correction-model forward after native reference construction.
- Energy, sensitivity, gradient, and exported descriptor are derived from the same validated descriptor values.
- Direct calls to `kernel`, `gradient`, and `descriptor` remain independently safe when no outer workflow context exists.
- A model or reference state change cannot reuse an earlier evaluation context.

### R2: Zero-Sensitivity Compact Fast Path

Priority: High.

Compact direct and compact Z-vector inference must bypass correction derivative construction and response solving when the validated descriptor sensitivity is exactly zero. The returned total gradient is the selected native gradient, and a coordinate-independent element constant affects energy only.

The fast path applies to compact inference results. Detailed calculations used to construct relaxed descriptor derivatives must continue to solve the response equations even when the method uses `model=None`, because force-data generation requires those derivatives for subsequent model training.

Acceptance criteria:

- `model=None` performs zero direct response solves and zero adjoint solves in compact inference.
- A supported model with an exactly zero local descriptor sensitivity uses the same fast path.
- The compact result equals the native gradient with exact shape, atom ordering, dtype, and finite-value checks.
- Detailed RHF force-data generation continues to publish explicit, response, and relaxed descriptor derivatives.

### R3: Consolidated Scientific-State and Integrity Utilities

Priority: High.

Array validation, immutable-array construction, stable value encoding, scientific-state fingerprints, response integrity fingerprints, and control validation must have canonical implementations shared by all reference families and the generic adjoint solver.

Hash updates for an already contiguous array must consume its buffer without creating another full `bytes` object. An unavoidable copy used to construct an immutable result must not be followed by an additional full copy solely for hashing.

The outer calculation transaction must calculate the complete state fingerprint only at the required entry and exit boundaries. Internal assertions must consume the transaction token without recalculating the fingerprint.

Acceptance criteria:

- The RHF public workflow performs no more than three complete scientific-state fingerprints including initial binding, transaction entry, and transaction exit.
- Nested gradient, descriptor, and response operations perform zero additional complete fingerprints.
- One canonical implementation owns array hashing and immutable-array construction.
- Existing mutation-detection and stale-response rejection tests continue to pass.
- New allocation tests detect a full-size transient `bytes` copy in response and adjoint hashing.

### R4: Shared Descriptor Derivative Workspace

Priority: High.

One derivative workspace must own the projected-density blocks, shell eigenvalue Jacobians, derivative projection overlaps, descriptor sensitivity contraction, and AO correction potential required by one calculation.

Compact and detailed paths must consume the same derivative primitives. Detailed mode may materialize additional published tensors, but it must not reconstruct projected-density blocks or eigenvalue Jacobians that already exist in the workspace.

Derivative projection overlaps depend on the bound geometry and projector and must be reused across spin components and derivative products within the same valid state.

Acceptance criteria:

- One detailed RHF gradient constructs each projected-density block and shell eigenvalue Jacobian once.
- UHF and UKS reuse derivative projection overlaps for both spin components.
- Compact execution does not materialize `dq/dP`, `dq/dR_response`, or full density partitions unless the selected backend requires them.
- Existing analytic-versus-finite-difference and direct-versus-Z-vector tolerances remain unchanged.

### R5: Atomic Energy Result Publication

Priority: Medium.

`DeePHF.kernel` must calculate and validate base energy, correction energy, and total energy in local values under one scientific-state transaction. It must publish `e_base`, `e_corr`, and `e_tot` only after the complete calculation succeeds.

Failure behavior must match gradient-driver behavior. A failed calculation must clear the result fields or expose a separately named last-successful result object; the three public fields must not silently look like the result of the failed call.

Acceptance criteria:

- A model-validation or non-finite-output failure cannot leave a newly partial energy state.
- The three energy fields are updated together after a successful calculation.
- Reference and model state are verified at transaction exit.
- Regression tests cover a successful call followed by a failing call.

### R6: Decompose PySCF Compatibility Adapters

Priority: High.

The PySCF compatibility layer must be decomposed by responsibility while preserving the existing public import paths through small compatibility facades.

The internal module boundaries must distinguish the following responsibilities:

- Reference validation, implementation identity, and immutable reference snapshots.
- Shared state, array, control, and fingerprint contracts.
- Restricted and unrestricted response problem construction.
- Direct response solution and contraction.
- Scalar-adjoint problem construction and solution.
- Native gradient integration.
- DFT functional and grid provenance.
- DFT grid-coordinate and grid-weight response.
- Explicit dense audit facilities used by bounded validation.

Recommended size constraints:

- A production module should remain at or below 1,200 physical lines and must remain below 1,500 physical lines.
- A calculation function should remain at or below 100 physical lines and must remain below 200 physical lines.
- A public compatibility facade should contain imports, public aliases, family assembly, and small validation entry points rather than solver implementations.

Acceptance criteria:

- `pyscf_rhf.py`, `pyscf_uhf.py`, `pyscf_rks.py`, and `pyscf_uks.py` become compatibility facades or family coordinators within the stated size constraints.
- DFT grid logic is isolated from generic response and adjoint algebra.
- Generic adjoint algebra remains independent of PySCF, model, workflow, and persistence modules.
- Public imports used by the current tests continue to resolve during the transition.

### R7: Shared Gradient-Driver Lifecycle

Priority: Medium.

Direct and Z-vector algorithms must remain independent, but their duplicated driver lifecycle must be centralized. The shared lifecycle includes binding validation, atom selection, result reset, scientific-state transaction entry, native-gradient acquisition, compact result validation, result publication, force conversion, and scanner adaptation.

Reference-family policies must provide family-specific response adapters, native-gradient functions, result error types, and optional detailed partitions. Exact reference types and family-specific scientific checks remain explicit.

Acceptance criteria:

- RHF, UHF, RKS, and UKS drivers use one lifecycle implementation for common control flow.
- Direct drivers cannot fall back to Z-vector solvers, and Z-vector drivers cannot fall back to direct response solvers.
- Family-specific policy code contains scientific differences rather than copied lifecycle code.
- The duplicated immutable-array, cycle-limit, version-series, atom-selection, and control-validation helpers are removed in favor of canonical owners.

### R8: Architecture Tests Must Protect Boundaries Rather Than File Placement

Priority: Medium.

Architecture tests must express allowed dependency directions, forbidden package dependencies, public facade behavior, direct/Z-vector independence, and generic-adjoint isolation. They must not require an implementation symbol to remain in one oversized physical file when the dependency boundary remains valid.

The current architecture test module must be split by objective so that dependency checks, public API checks, solver-independence checks, and source constraints can evolve independently.

Acceptance criteria:

- No architecture test contains a complete mapping of implementation symbols to the four oversized PySCF adapter files.
- Dependency rules continue to reject PySCF imports from method-neutral descriptor and adjoint algebra.
- Direct/Z-vector independence is tested through calls and dependencies rather than duplicated source text.
- Each architecture test module remains below 500 physical lines.

### R9: Performance Regression Budgets

Priority: High.

The test suite must include deterministic operation-count budgets for the public workflow and detailed force-data path. Wall-clock thresholds may supplement these budgets on dedicated benchmarks but must not replace them in the normal suite.

Required counters include descriptor evaluations, model forwards, state fingerprints, derivative-overlap integral evaluations, projected-density construction, shell eigenvalue Jacobian construction, direct response solves, adjoint solves, full density-partition materializations, response operator actions, and preconditioner actions.

Acceptance criteria:

- Tests fail when the R1 through R4 operation budgets regress.
- Compact and detailed backends have separate budgets.
- Zero-sensitivity and nonzero-sensitivity models have separate budgets.
- Bounded peak-memory checks cover response-result hashing and detailed derivative construction.
- Scientific accuracy assertions remain independent from performance assertions so that a failure reports the violated contract precisely.

## Implementation Sequence

1. Add operation counters and regression tests that reproduce the current redundant work without changing numerical behavior.
2. Make energy publication atomic and establish the scoped evaluation context.
3. Add the zero-sensitivity compact fast path.
4. Introduce canonical state, array, fingerprint, and control utilities.
5. Introduce the shared descriptor derivative workspace and migrate force-data generation.
6. Extract the common gradient-driver lifecycle while preserving separate direct and Z-vector solver dependencies.
7. Split the PySCF compatibility adapters and replace file-placement architecture tests with dependency-boundary tests.
8. Run objective-specific analytic-force suites, the response-scalability suite, force-training tests, and the complete locked test suite.

## Verification Commands

```bash
uv sync --locked --python 3.11
uv run pytest tests/analytic_forces
uv run pytest tests/zvector_inference
uv run pytest tests/uhf_analytic_forces tests/uhf_zvector_inference
uv run pytest tests/rks_analytic_forces tests/rks_zvector_inference
uv run pytest tests/uks_analytic_forces tests/uks_zvector_inference
uv run pytest tests/response_scalability tests/force_training
uv run pytest
```

## Completion Criteria

The change set is complete when all supported DeePHF scientific results retain their current numerical tolerances, the public workflow and zero-sensitivity budgets pass, detailed derivative construction meets the shared-workspace budgets, energy result publication is atomic, every production source file satisfies the size constraints, architecture tests enforce dependency boundaries without locking implementation placement, and the complete locked Python 3.11 suite passes from a clean worktree.
