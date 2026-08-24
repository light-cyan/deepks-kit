# DeePHF Change Requirements 3

## Purpose

This document defines the next corrective refactor for the perturbative DeePHF calculation layer. The work must close the remaining scientific-state cache defect, remove redundant state scans from controlled workflows, reduce module fragmentation without recreating oversized program files, narrow compatibility facades, and finish the remaining mechanical consolidation.

## Counting Definitions

A Python file is any recursively discovered path with the `.py` suffix under `deepks/deephf/`, including package initializers, construction modules, and the four compatibility facades. A top-level file is one of those paths whose parent is exactly `deepks/deephf/`; an audit file is one whose parent is `deepks/deephf/audits/`.

A physical line is one element of `Path.read_text(encoding="utf-8").splitlines()`, including blank, comment, facade, and construction lines. Function size is the inclusive AST source span from the earliest decorator or definition line through `end_lineno`.

A complete scientific-state fingerprint hashes the reference, descriptor, complete model execution state, and device. A cache-state fingerprint is the conservative reference-and-descriptor hash used only before a calculation-scoped cached value can be reused. A cheap evidence validation compares identities, execution configuration, tensor metadata, storage identity, and mutation versions without hashing full tensor buffers. A cache hit is reuse of an already materialized numerical value inside one evaluation context.

Operation reports cover one calculation transaction and exclude construction. Method construction performs one complete binding fingerprint separately. A controlled workflow is a package-owned uninterrupted orchestration such as `evaluate_molecule`; a public workflow inside `calculation()` is user-interruptible and validates state before a later public boundary can consume cached values.

## Scope

The requirements cover `deepks/deephf/`, the CorrNet execution-state facilities in `deepks/model/model.py`, the descriptor derivative workspace consumed by DeePHF, architecture tests, deterministic operation budgets, and the RHF, UHF, RKS, and UKS scientific regression suites.

The supported calculation surface remains RHF, UHF, RKS, and UKS references; direct and scalar-adjoint analytic gradients; compact energy, descriptor, gradient, and force inference; detailed RHF force-data generation; atom selection; supported scanners; strict reference validation; and deterministic DFT grid handling.

## Current Baseline

Revision `0a1b6f5` provides calculation-scoped cache ownership, owned immutable public numerical results, shared descriptor derivative primitives, zero-sensitivity compact bypasses, transaction-wide result-publisher cleanup, lazy model-output validation, recursive source-size checks, small PySCF compatibility facades, and operation counters for cache and solver work.

The Python 3.11 suite contains 856 passing tests. The architecture and deterministic DeePHF performance objectives contain 59 passing tests. Every function under `deepks/deephf/` remains below the 200-line hard limit, and the largest current module contains 908 physical lines.

`deepks/deephf/` currently contains 58 Python files and 17,637 physical lines. Forty-two files are at package top level, 16 files are under `audits/`, 26 files contain at most 200 lines, and each non-empty audit implementation module has one production importer. The package is therefore within individual source-size limits while remaining excessively fragmented.

The active cache-state guard remains incomplete for model execution graph mutations. `model_state_evidence` and `model_state_fingerprint` cover module types, selected call identities, tensor metadata, tensor versions, and tensor values, but they do not bind every plain Python attribute that changes CorrNet execution, including the selected activation and residual policy. A supported activation change inside one calculation transaction can therefore reuse an earlier energy and sensitivity without invalidating the transaction.

The normal energy-gradient-descriptor workflow performs two complete transaction fingerprints, three conservative cache-state fingerprints, and three state-version validations after method construction. The first public boundary performs these cache checks before any reusable numerical cache exists.

## Required Changes

### R1: Establish One Canonical Model Execution-State Contract

Priority: Critical.

One canonical implementation must describe every model property that can change a DeePHF correction energy, descriptor sensitivity, or force result. DeePHF cache validation, scanner validation, force-model validation, checkpoint validation, and force-aware training must consume this canonical contract instead of maintaining partially overlapping evidence functions.

For the supported force model, the execution-state contract must cover the exact CorrNet type, module identities and types, trusted call dispatch, class and instance `forward` implementations, compilation hooks, module hooks, global hooks, training modes, `input_dim`, projector metadata, element metadata, embedder identity and execution configuration, DenseNet activation identity, residual policy, residual scaling configuration, layer topology, parameter and buffer identity, storage identity, shape, layout, dtype, device, `requires_grad`, mutation version, and numerical value when a complete fingerprint is required.

`force_model_structure_evidence` in `deepks/model/model.py` should become the starting owner for the cheap graph evidence contract. It must be extended to include every execution-affecting field, including `DenseNet.use_resnet` and mutable embedder configuration, before DeePHF adopts it. `model_state_evidence` in `deepks/deephf/capabilities.py` must then delegate to the canonical owner or be removed.

Generic `torch.nn.Module` models may remain valid for energy evaluation under the existing scalar-output contract. A generic model whose complete mutable execution state cannot be proven must not reuse a cached model output across public boundaries without a conservative model fingerprint. Force evaluation remains restricted to the exact validated CorrNet graph.

Acceptance criteria:

- Changing `densenet.actv_fn` between two public calls in one transaction is detected before the second call returns a cached result.
- Changing `densenet.use_resnet`, residual scaling configuration, embedder configuration, execution hooks, compilation dispatch, model mode, input metadata, projector metadata, element metadata, or a trusted implementation identity is detected at the first dependent boundary.
- Replacing a parameter, buffer, layer, embedder, or model with an equivalent-looking object cannot retain a stale cache solely because its type and numerical values match.
- Parameter and buffer value mutations are detected before cached energy or sensitivity reuse.
- One canonical graph-evidence function and one canonical complete model-fingerprint function own the supported CorrNet state definitions.
- Scanner, force inference, force-aware training, and transaction cache tests use the same canonical evidence contract.
- Tests reproduce a nonzero stale-energy and stale-gradient difference before the fix and verify fail-closed behavior after the fix.

### R2: Make Built-In Model Mutations Version-Aware

Priority: Critical.

CorrNet mutation methods must not write parameters through `.data`. `set_normalization`, `set_prefitting`, and `set_energy_const` must update existing parameters under `torch.no_grad()` through operations that advance the Torch mutation version, preserve registration, preserve dtype and device, and retain the declared trainability contract.

Any model method that replaces storage must either be removed in favor of an in-place tracked update or explicitly invalidate every bound DeePHF calculation context before the new state can be consumed.

Acceptance criteria:

- Every built-in model setter advances the affected tensor version when it changes a value.
- Setter calls preserve parameter object identity unless the public setter contract explicitly requires replacement.
- Calling any built-in setter between public calls in one calculation transaction fails before stale cache reuse.
- Setter calls outside an active transaction cause the next independent calculation to use the updated values.
- Tests cover normalization, prefitting weight and bias, energy constant, trainability, dtype, and device preservation.
- No production CorrNet mutation uses `.data` assignment or slicing.

### R3: Separate Controlled Workflow Reuse from User-Interruptible Transactions

Priority: High.

The internal energy-gradient-descriptor workflow must have a trusted orchestration path that does not return control to user code between its component calculations. That path may reuse one accepted state token and one evaluation context without conservatively rehashing the unchanged reference and descriptor before each internal component.

The public `calculation()` context remains user-interruptible and must continue to detect mutations before cached reuse at each public boundary. Its additional safety cost must be reported separately from the controlled workflow budget.

A cache-state check must run only when the context already contains a cached value that the next operation can consume. The first public calculation in a fresh context must not perform a conservative cache fingerprint solely to confirm the validity of an empty cache.

Cheap state evidence and conservative fingerprints must have distinct ownership and counters. Cheap evidence should reject trusted versioned mutations without proceeding to a full hash. Conservative reference or descriptor fingerprints should run only for state that lacks reliable mutation versions and only at an actual reuse boundary.

Acceptance criteria:

- One controlled `evaluate_molecule` workflow performs no more than three complete scientific-state fingerprints including initial method binding, transaction entry, and transaction exit.
- The controlled workflow performs zero intermediate conservative cache-state fingerprints while retaining one descriptor evaluation and one model forward.
- A fresh public context performs no cache-state fingerprint before its first cache-producing calculation.
- A later public call in a user-interruptible context detects in-place NumPy reference mutations before returning a dependent cache value.
- Cheap model evidence rejects ordinary tracked model mutations without hashing complete parameter buffers.
- Conservative hashes remain buffer-based and do not materialize transient full-size `bytes` objects.
- Performance tests express maximum budgets rather than requiring redundant work as an exact positive count.

### R4: Reduce the DeePHF Module Count Without Recreating Large Files

Priority: High.

The package must be consolidated along algorithm and dependency boundaries rather than the current Cartesian product of reference family, backend, response concern, and audit concern. Direct and scalar-adjoint algorithms remain physically separate, while family implementations of the same backend may share one module.

The preferred post-refactor target is 38 to 42 Python files under `deepks/deephf/`. The conservative target is at most 42 files while keeping every normal implementation module at or below 800 physical lines where practical and every module below 1,000 physical lines. `pyscf_dft_provenance.py` may remain near its current size while its cohesive provenance responsibility remains unchanged.

The following consolidation topology is recommended:

| Current group | Current files | Current physical lines | Target |
| --- | ---: | ---: | --- |
| RHF, RKS, UHF, and UKS direct gradient drivers | 4 | 757 | One direct-gradient module |
| RHF, RKS, UHF, and UKS scalar-adjoint gradient drivers | 4 | 643 | One scalar-adjoint-gradient module |
| Audit package including its initializer | 16 | 4,098 | Nine files including the initializer |
| UHF and UKS method modules | 2 | 358 | One unrestricted-method module |
| UHF and UKS reference modules | 2 | 790 | One unrestricted-reference module |
| Generic scanner and RHF scanner implementation | 2 | 692 | One scanner module |
| RKS reference and native-gradient modules | 2 | 295 | One RKS reference module |

The audit package should group reference checks into restricted and unrestricted modules, group operator checks with the response audit that consumes them, group UHF and UKS response audits where the UKS contract composes the unrestricted implementation, and group UHF and UKS adjoint audits. RHF and RKS adjoint audits may remain separate to keep each module comfortably below the preferred size.

A new private module is justified when it protects a hard dependency direction, preserves direct versus scalar-adjoint independence, has multiple production consumers, or contains a cohesive implementation that would push its owner beyond the module-size limit. A private single-consumer module below 200 lines should otherwise be merged into its consumer or a cohesive shared owner.

Acceptance criteria:

- `deepks/deephf/` contains at most 42 Python files after consolidation.
- The four public `pyscf_rhf.py`, `pyscf_uhf.py`, `pyscf_rks.py`, and `pyscf_uks.py` compatibility paths remain importable.
- Direct gradient drivers and scalar-adjoint gradient drivers remain in different physical modules and do not call one another.
- Generic scalar-adjoint algebra remains independent of PySCF, model, workflow, and persistence modules.
- Dense audits remain lazily imported and absent from compact production imports.
- Every function remains below 200 physical lines, and orchestration functions remain at or below 100 physical lines.
- No consolidated module exceeds 1,000 physical lines.
- Architecture tests protect dependency and algorithm boundaries without mapping each implementation symbol to a specific file.

### R5: Narrow the PySCF Compatibility Facades

Priority: High.

The four PySCF compatibility facades must expose an explicit supported public surface. Wildcard imports, private implementation aliases, imported dependency modules, mutable caches, numerical tolerances, solver core classes, and test-only patch points must not become facade attributes.

Tests that inspect or patch an internal implementation must import the implementation owner directly. Public facade tests must validate only documented public symbols and a deliberate `__all__`.

Acceptance criteria:

- Every compatibility facade defines an explicit `__all__`.
- Compatibility facades contain no wildcard import.
- Compatibility facades contain no explicit alias whose exported name begins with an underscore.
- `from deepks.deephf.pyscf_<family> import *` exports only the supported public surface.
- Scientific tests patch response cores, integrity helpers, and tolerances through their owner modules rather than a facade.
- Internal production modules continue to avoid importing the public facades.
- Public names already re-exported by `deepks.deephf` continue to resolve unless a separately approved API change removes them.

### R6: Finish Mechanical Consolidation

Priority: Medium.

Family classes must inherit unchanged constructor behavior instead of repeating pass-through `__init__` methods. A subclass should define a constructor only when it validates, transforms, stores, or documents a genuinely family-specific argument contract.

Restricted response algebra must have one owner for density-from-MO-response, MO-potential, and induced-potential contractions. A subclass must not override a mixin method solely to call the same shared function with unchanged arguments.

Identical scanner construction, atom-index normalization, response atom selection, residual statistics, cycle limits, scalar control validation, and exact operator-validation scaffolding must use canonical helpers when their semantics and error domains match. Family-specific scientific errors and tolerances remain explicit policies rather than being hidden behind generic string substitution.

Acceptance criteria:

- RKS, UHF, and UKS method classes contain no unchanged pass-through constructor.
- The RKS restricted response core inherits the canonical density-response contraction without an equivalent wrapper override.
- RHF direct and scalar-adjoint scanner construction has one owner.
- Atom-selection and scalar-control helpers have one canonical owner for each actual contract.
- Exact-AST duplicate detection reports no production duplicate longer than four statements unless an approved scientific-policy exception is documented beside both implementations.
- Consolidation reduces code or ownership ambiguity and does not introduce a new abstraction used by only one caller.

### R7: Expand Correctness, Efficiency, and Topology Regression Tests

Priority: High.

Scientific correctness tests, state-safety tests, operation budgets, allocation budgets, dependency checks, public API checks, and topology checks must remain separate objectives so that each failure identifies the violated contract.

State-safety tests must mutate every supported category of model execution state before a cached public result is consumed. They must compare against a fresh recomputation and must verify both fail-closed behavior and complete result cleanup.

Topology tests should enforce aggregate limits and dependency rules without reintroducing a complete symbol-to-file placement map. A small inventory helper may report file counts, line distributions, single-consumer private modules, wildcard imports, and private facade exports.

Acceptance criteria:

- State tests cover activation, residual policy, embedder configuration, hooks, built-in setters, conventional `no_grad` tensor mutation, storage replacement, dtype, device, mode, and reference-array mutation.
- At least one state mutation test proves that a previously stale nonzero energy and gradient are rejected before return.
- RHF, UHF, RKS, and UKS retain direct and scalar-adjoint zero- and nonzero-sensitivity budgets.
- Controlled and user-interruptible workflows have separate fingerprint budgets.
- Architecture tests fail when the package exceeds the agreed module-count or module-size limits.
- Facade tests fail on wildcard imports, private aliases, or missing explicit public exports.
- Finite-difference and direct-versus-adjoint tolerances remain unchanged.
- The complete locked Python 3.11 suite passes.

### R8: Keep Current Documentation Quantitatively Accurate

Priority: Medium.

The current DeePHF requirements and completion report must use the same definitions for a Python file, physical line, complete fingerprint, cache-state fingerprint, cheap evidence validation, cache hit, and public workflow. Counts must identify whether construction and compatibility facades are included.

The current acceptance description must not state that every model mutation is covered until execution graph, setter, and generic-model cache tests pass. Structural completion must report both maximum file size and total module count.

Acceptance criteria:

- The completion report states the final Python file count, top-level file count, audit file count, total physical lines, maximum module size, and maximum function size.
- Fingerprint reports distinguish construction, controlled workflow, public user-interruptible workflow, cheap evidence, and conservative hashes.
- Current documentation names only behavior protected by passing tests or direct verification.
- Archived requirements and project audits remain under `docs/legacy/` and are not used as current implementation specifications.

## Implementation Order

1. Add failing activation, residual-policy, hook, setter, and generic-model cache tests that reproduce stale reuse before changing production code.
2. Establish and adopt the canonical model execution-state evidence and fingerprint contracts.
3. Replace `.data` mutations with tracked CorrNet setter operations.
4. Separate the controlled internal workflow from the user-interruptible calculation context and revise fingerprint budgets.
5. Narrow compatibility facades and move private test imports to implementation owners.
6. Consolidate direct and scalar-adjoint driver modules.
7. Consolidate audit, unrestricted method/reference, scanner, and RKS native-gradient modules to the target topology.
8. Remove remaining exact mechanical duplicates and unused compatibility aliases.
9. Run focused state, performance, architecture, and scientific tests followed by the complete suite.

## Verification

Run the focused state, structure, and performance objectives first:

```bash
uv run pytest tests/deephf_performance
uv run pytest tests/architecture
uv run pytest tests/analytic_forces/test_deephf_strict_contract.py
uv run pytest tests/baseline/test_deephf_public_workflow.py
```

Run every analytic-force and scalar-adjoint family afterward:

```bash
uv run pytest tests/analytic_forces tests/zvector_inference
uv run pytest tests/rks_analytic_forces tests/rks_zvector_inference
uv run pytest tests/uhf_analytic_forces tests/uhf_zvector_inference
uv run pytest tests/uks_analytic_forces tests/uks_zvector_inference
```

Run the complete locked environment verification last:

```bash
uv sync --locked --python 3.11
uv run pytest
```

For public export, module layout, or packaging changes, also run:

```bash
uv build
```
