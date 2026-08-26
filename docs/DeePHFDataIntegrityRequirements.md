# DeePHF Data Integrity Requirements

## Baseline

This work starts from revision `6f3f9b14a4880391cbc86c9ae4c104f329087c5a`, whose scientific validation experiments characterize the current RHF, UHF, RKS, and UKS inference paths. The force-training contract remains RHF-specific.

## Purpose

Two silent failure modes require explicit contracts. Persisted force labels must identify the target calculation that supplied the supervised energy and force in addition to the RHF baseline and response calculation, while multi-frame CLI inference must prove that adjacent frames retain the same electronic root.

These safeguards require machine-readable treatment because both failure modes can produce numerically finite outputs that pass local equation checks while representing incompatible potential-energy surfaces. Manual directory naming and operator judgment are useful review aids but are not reliable compatibility contracts for grouped training, checkpoint reuse, or automated trajectory processing.

## Design Constraints

The implementation must add one target-identity value owned by the existing force schema and one family-neutral occupied-subspace comparison owned by the DeePHF reference workflow. Method-specific force producers, readers, trainers, checkpoints, and inference code must consume these owners instead of defining parallel metadata or overlap rules.

The change must preserve the current support boundary: analytic-gradient inference supports RHF, UHF, RKS, and UKS, while persisted relaxed-force data and force-aware training remain RHF-specific. Electronic-root tracking must not require four duplicated geometry scanners.

Single-frame Python inference remains stateless. Each system passed to the multi-frame CLI is treated as one ordered sequence, and state is reset between systems.

## Target Identity Contract

The current persisted RHF force-data schema is version `2`. Every dataset must contain one canonical target identity shared by all frames. The required fields are `method`, `basis`, `software`, `version`, `frozen_core`, `relativistic`, `state`, `energy_force_consistent`, and `settings`.

```json
{
  "method": "CCSD(T)",
  "basis": "cc-pVTZ",
  "software": "target-code",
  "version": "1.0",
  "frozen_core": true,
  "relativistic": "none",
  "state": "closed-shell singlet ground state",
  "energy_force_consistent": true,
  "settings": {"energy_tolerance": 1e-10}
}
```

The string fields must be nonempty, `frozen_core` and `energy_force_consistent` must be booleans, and `settings` must be a canonical JSON mapping for convergence thresholds and method-specific controls. Strict force training requires `energy_force_consistent=true` because energy and force labels must describe one differentiable target surface.

The complete normalized target identity must participate in the dataset compatibility fingerprint. It must also be retained, together with its own fingerprint, in force-training checkpoint metadata and CLI inference provenance when inference loads a force-trained checkpoint.

Grouped readers must reject datasets whose target identities differ through their existing compatibility-fingerprint comparison. No second target-comparison path is needed.

## Electronic-Root Continuity Contract

The multi-frame CLI must use the previous converged density matrix as the next frame's SCF initial guess. After convergence, it must compare the previous and candidate occupied subspaces with the cross-geometry AO overlap matrix.

Restricted references have one occupied-space overlap. Unrestricted references have independent alpha and beta occupied-space overlaps, and an empty spin channel contributes the neutral overlap value `1.0`. The minimum singular value across populated channels is the continuity diagnostic.

A candidate below the configured overlap tolerance must be rejected before its energy, descriptor, or gradient is published. A failed candidate must not advance the accepted density or root anchor.

Each successful system output must contain one atomic JSON provenance file with the reference family, overlap tolerance, per-frame state fingerprint, parent-state fingerprint, initial-guess source, spin-resolved occupied-space overlaps, and minimum occupied overlap. Force-trained model target identity must be included in the same inference provenance when available.

The existing RHF scanner must reuse the family-neutral overlap calculation while retaining its stricter object-lifetime and publication guarantees.

## Acceptance Criteria

- A strict force dataset without target identity is rejected.
- Changing any target identity field changes dataset compatibility and causes grouped-reader rejection.
- Force-training checkpoint metadata preserves the normalized target identity and rejects a different target contract.
- CLI inference persists checkpoint target identity when the model contains force-training metadata.
- The first frame of each CLI system uses an independent initial guess, and every later frame uses the preceding accepted density matrix.
- RHF, UHF, RKS, and UKS occupied-subspace continuity uses one shared numerical implementation.
- A discontinuous candidate is rejected without publishing partial system output or advancing the root anchor.
- Per-frame root provenance is persisted for successful multi-frame CLI output.
- Existing single-frame inference, analytic gradients, RHF scanner behavior, energy-only training, and RHF force-training numerics remain unchanged apart from the stricter persisted metadata contract.

## Verification

Run the focused data and trajectory tests first, followed by the baseline and complete suites:

```bash
uv run pytest tests/force_training tests/trajectory_safety
uv run pytest tests/baseline
uv run pytest
```

For the schema and checkpoint format change, also run:

```bash
uv sync --locked --python 3.11
uv build
```
