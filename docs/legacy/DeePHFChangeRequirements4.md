# DeePHF Change Requirements 4

## Purpose

This document defines a focused correction for the remaining DeePHF cache-safety and source-consolidation defects. The implementation should favor a small, explicit solution over additional state machinery, module splitting, or broad architectural redesign.

## Current Baseline

Revision `a875f92` contains 41 Python files under `deepks/deephf/`, including 32 top-level files and nine audit files. The complete Python 3.11 suite contains 885 passing tests, and source and wheel builds succeed.

The current implementation correctly detects tracked tensor mutations, parameter replacement, model mode changes, activation changes, hooks, residual configuration, and supported metadata changes. Controlled package workflows also retain the intended single descriptor evaluation and single model forward.

Two cache-safety gaps remain in the public user-interruptible calculation context:

- A parameter or buffer changed through an untracked storage alias such as `.data` or a shared NumPy view can retain the same tensor identity, storage identity, and Torch mutation version. A public call can therefore return a cached model result before the complete fingerprint at transaction exit detects the change.
- `ThermalEmbedding.forward` depends on module-level helper functions including `pad_masked`, `masked_softmax`, and `unpad_masked`, but the canonical graph evidence records the `forward` identity without recording those helper identities. Replacing one of these helpers can change a fresh result while leaving both the cheap evidence and complete model fingerprint unchanged.

The module-count consolidation also retained mechanical remnants from the original files. Several merged modules contain repeated imports, intermediate string literals that previously served as module docstrings, overwritten `__all__` assignments, and unchanged pass-through constructors.

## Scope

The work covers the canonical CorrNet execution-state contract in `deepks/model/model.py`, public DeePHF cache validation in `deepks/deephf/`, focused state-safety tests, and mechanical cleanup of the already consolidated DeePHF modules.

The existing RHF, UHF, RKS, and UKS scientific algorithms, direct and scalar-adjoint separation, public compatibility facades, module layout, and controlled workflow behavior remain the implementation baseline.

## Required Changes

### R1: Validate Complete Model Values Before Public Cache Reuse

Priority: Critical.

The public user-interruptible `calculation()` path must validate the complete model fingerprint before returning a cached model-dependent value. Cheap evidence may run first so ordinary tracked mutations fail immediately, but equality of cheap evidence must not be treated as proof that tensor values are unchanged.

The complete fingerprint must hash current parameter and buffer values and must therefore detect changes performed through `.data`, shared NumPy views, or other aliases that do not advance `Tensor._version`.

The controlled package-owned workflow remains uninterrupted and may continue to reuse its accepted transaction state without intermediate complete model hashes. This preserves the normal inference performance path while giving the public interruptible path a simple correctness rule.

Acceptance criteria:

- A `.data` parameter mutation between two public calls raises `DeePHFCapabilityError` before the second call returns a cached result.
- A parameter or buffer mutation through a writable shared NumPy view is rejected at the same boundary.
- The rejected transaction clears energy and gradient publisher results.
- A fresh independent calculation after the mutation uses the new parameter or buffer values.
- Ordinary tracked mutations continue to fail through cheap evidence without performing an unnecessary fallback hash after the mismatch is known.
- Controlled `evaluate_molecule` and force-data workflows retain one descriptor evaluation and one model forward.

### R2: Bind Trusted CorrNet Helper Implementations

Priority: Critical.

The canonical CorrNet graph evidence must include every project-owned helper implementation directly executed by the supported `CorrNet`, `DenseNet`, `TraceEmbedding`, and `ThermalEmbedding` forward paths. At minimum, the evidence must bind `pad_masked`, `masked_softmax`, and `unpad_masked`, together with any other mutable module-level callable whose replacement can change the supported model result.

The helper contract should be one explicit canonical tuple owned beside `force_model_structure_evidence`. Force validation, complete model fingerprinting, cache validation, checkpoint validation, and force-aware training must continue to consume that same owner.

Acceptance criteria:

- Replacing each trusted embedding helper between public calls is detected before cached energy or sensitivity is returned.
- The complete model fingerprint changes when a trusted helper implementation changes.
- Restoring the original helper restores the original fingerprint for an otherwise unchanged model.
- Tests use a nontrivial `ThermalEmbedding` result and prove that a fresh recomputation would differ from the cached value.
- Supported models without an embedder and models using `TraceEmbedding` retain their current numerical behavior.

### R3: Finish the Existing Module Consolidation

Priority: High.

The existing 41-file layout should be retained. Cleanup should occur inside the consolidated owners and should not split cohesive numerical implementations merely to satisfy a source-length target.

The following consolidated modules require mechanical cleanup:

- `gradient.py`
- `zvector.py`
- `scanner.py`
- `pyscf_rks_reference.py`
- `unrestricted_method.py`
- `unrestricted_reference.py`
- `audits/restricted_reference.py`
- `audits/unrestricted_reference.py`
- `audits/rhf_response_audit.py`
- `audits/rks_response_audit.py`
- `audits/unrestricted_adjoint.py`
- `audits/unrestricted_response.py`

Each module must have one module docstring, one import section, and at most one final `__all__` assignment. Imports needed by multiple family implementations must be owned once at module scope.

The RHF, RKS, UHF, and UKS classes in `gradient.py` and `zvector.py` must inherit the unchanged `GradientDriver` constructor instead of repeating pass-through `__init__` methods.

Acceptance criteria:

- No consolidated module contains a standalone top-level string literal after its module docstring.
- No consolidated module imports the same symbol more than once.
- No consolidated module assigns `__all__` more than once.
- Direct-gradient and scalar-adjoint classes contain no unchanged pass-through constructor.
- The package remains at or below 41 Python files.
- Direct and scalar-adjoint implementations remain in separate modules and do not call one another.
- Audit modules remain lazily imported by production code.
- Public exports and scientific numerical behavior remain unchanged.

### R4: Add Focused Regression Protection

Priority: High.

Tests should protect the defects addressed by this change without encoding a complete symbol-to-file map or introducing new source-length classifications.

State tests must distinguish detection before a cached call returns from detection at transaction exit. A test that observes only the final context-manager exception is insufficient for public cache safety.

Architecture tests should inspect the consolidated modules for repeated imports, multiple `__all__` assignments, intermediate top-level string literals, and pass-through family constructors.

Acceptance criteria:

- State tests cover tracked mutation, `.data` mutation, NumPy-alias mutation, parameter replacement, and trusted helper replacement.
- Each untracked mutation test verifies a nonzero difference against a fresh recomputation.
- Each mutation test verifies complete result cleanup.
- Architecture tests fail when any listed mechanical consolidation remnant is reintroduced.
- Existing finite-difference and direct-versus-adjoint tolerances remain unchanged.
- The complete locked Python 3.11 suite and package build pass.

## Implementation Order

1. Add failing tests for `.data`, NumPy-alias, and trusted-helper mutations.
2. Add complete model-value validation at actual public cache-reuse boundaries.
3. Extend the canonical CorrNet graph evidence with trusted helper identities.
4. Clean the existing consolidated modules in place.
5. Add small architecture checks for the mechanical remnants.
6. Run focused state and architecture tests, followed by the complete suite and build.

## Verification

Run the focused objectives first:

```bash
uv run pytest tests/deephf_state_safety
uv run pytest tests/deephf_performance
uv run pytest tests/architecture
```

Run the scientific regression families:

```bash
uv run pytest tests/analytic_forces tests/zvector_inference
uv run pytest tests/rks_analytic_forces tests/rks_zvector_inference
uv run pytest tests/uhf_analytic_forces tests/uhf_zvector_inference
uv run pytest tests/uks_analytic_forces tests/uks_zvector_inference
```

Run final verification:

```bash
uv sync --locked --python 3.11
uv run pytest
uv build
git diff --check
```
