# P3A RHF Relaxed-Force Data and Training

## 1. Status and scope

P3A is complete for the strict molecular RHF domain accepted by the [P2 direct oracle](./p2_rhf_direct_oracle.md). It connects validated native RHF response calculations to persistent force data, strict readers, energy-and-force training, validation, saved-data testing, and checkpoint reload.

The scientific input is a native converged PySCF RHF reference inside the P2 support domain, a fixed normalized projector basis, and real `float64` target energies and forces. The persistent training derivative is the complete `dq_dR_relaxed` produced by the P2 direct response.

The [P0 scientific contract](./p0_scientific_contract.md) remains authoritative for DeePHF method meaning, method-package separation, derivative terminology, and the distinction between perturbative DeePHF and variational DeePKS.

## 2. Strict persisted data contract

Schema v1 has identity `deepks.deephf.rhf-force-data`, version `1`, and manifest filename `force_data.json`. A dataset directory contains the manifest and exactly the following nine canonical NumPy arrays described by that manifest.

| Field | Axes and exact shape | Unit | Sign | Meaning |
|---|---|---|---|---|
| `atom` | `(frame, raw_atom, 4)` | nuclear charge plus `Bohr` | not applicable | Nuclear charge followed by the raw-atom Cartesian position. |
| `descriptor` | `(frame, descriptor_atom, feature)` | `1` | not applicable | Ordered projected-density eigenvalues. |
| `e_base` | `(frame, 1)` | `Eh` | `+energy` | Native RHF energy. |
| `f_base` | `(frame, raw_atom, 3)` | `Eh/Bohr` | `force=-dE/dR` | Native RHF force. |
| `e_target` | `(frame, 1)` | `Eh` | `+energy` | Supervised total energy. |
| `f_target` | `(frame, raw_atom, 3)` | `Eh/Bohr` | `force=-dE/dR` | Supervised total force. |
| `e_corr_target` | `(frame, 1)` | `Eh` | `+energy` | Supervised correction energy. |
| `f_corr_target` | `(frame, raw_atom, 3)` | `Eh/Bohr` | `force=-dE/dR` | Supervised correction force. |
| `dq_dR_relaxed` | `(frame, raw_atom, cartesian, descriptor_atom, feature)` | `Bohr^-1` | `+dq/dR` | Complete descriptor derivative including explicit motion and validated RHF reference response. |

Every array is real, finite, C-contiguous `numpy.float64`; every declared dimension is positive; the Cartesian order is `x, y, z`; and the energy, force, length, and Jacobian units are fixed by the manifest conventions. Schema v1 requires one descriptor atom for every raw atom, records both mapping directions, and requires positive integer nuclear charges with `ghost_policy: rejected`.

The official RHF producer enforces `e_corr_target = e_target - e_base` and `f_corr_target = f_target - f_base` to a maximum absolute residual of `1e-12`. It rejects a missing canonical array, an additional array name or file, a wrong dtype, a nonfinite value, a wrong axis order or shape, or inconsistent derived targets before writing a valid manifest.

`deepks.deephf.write_rhf_force_dataset(...)` is the public persistence entry point and installs the manifest only after every RHF frame has passed the direct-response producer. Its method-neutral low-level writer is internal to the producer. `deepks.data.load_force_dataset(directory)` verifies the schema identity, exact manifest structure, conventions, dimensions, field contracts, array hashes, provenance, derived-target identities, and manifest fingerprint before returning a sealed `ForceDataContract` and the arrays.

## 3. Derivative and force semantics

For raw atom `A`, Cartesian component `x`, descriptor atom `I`, and feature `k`, the P2 oracle defines `dq_dR_relaxed[A,x,I,k] = dq_dR_explicit[A,x,I,k] + dq_dR_response[A,x,I,k]`.

For a correction model `e_corr_theta(q)`, `deepks.model.evaluate.predict_correction` evaluates the complete model sensitivity `partial e_corr_theta / partial q[I,k]`, including normalization and every active model branch, and contracts it as `f_corr_theta[A,x] = -sum_I,k (partial e_corr_theta / partial q[I,k]) dq_dR_relaxed[A,x,I,k]`.

The force label uses the same sign: `f_base = -partial e_base/partial R` and `f_corr_target = f_target - f_base`. The model therefore learns the correction energy and correction force under one consistent atomic-unit convention.

The runtime gradient field registry keeps `dq_dR_explicit`, `dq_dR_response`, and `dq_dR_relaxed` distinguishable for diagnostics. The strict persisted training contract carries the complete relaxed derivative as `dq_dR_relaxed`.

## 4. Provenance, hashes, and compatibility

The manifest records the complete current provenance needed to interpret each array:

- `atom_mapping` records `descriptor_to_raw`, `raw_to_descriptor`, nuclear charges, and the ghost policy.
- `descriptor` records the canonical ordered-spectrum definition, spin-summed semantics, shell sizes, normalized projector-basis content and SHA-256 digest, and the numerical differentiability controls used by the producer.
- `reference` records the exact RHF family and Python class, basis content and SHA-256 digest, ECP state, charge, spin, closed-shell occupations, and SCF controls.
- `response` records the `rhf_direct` backend, the `deepks.deephf.pyscf_rhf.RHFResponseAdapter` identity, and response controls.
- Each frame records its RHF state and response-integrity fingerprints, geometry, reference and response convergence state, complete response diagnostics and controls, descriptor diagnostics, per-field frame hashes, and a derived sample ID.
- `generation` records the accepted producer identity and version and the DeepKS, PySCF 2.14, PyTorch, NumPy, and Python versions together with the available DeepKS commit identity.

Each field entry contains a SHA-256 digest over its dtype, shape, and bytes. Each frame sample ID binds the compatibility fingerprint, atom mapping, occupations, frame provenance, and the hashes of its nine frame slices. The manifest fingerprint binds the canonical complete manifest.

The compatibility fingerprint identifies cross-system training compatibility rather than frame identity. It binds schema and tensor conventions, field semantics, `float64`, the complete descriptor and differentiability interpretation, shell sizes, projector digest, feature count, RHF family and class, basis, ECP state, charge, spin, SCF controls, complete direct-response controls, producer identity, and generation versions. Compatible datasets may have different frame counts, atom counts, nuclear compositions, occupations, geometries, values, and sample IDs while preserving those shared scientific controls.

`ForceDataContract` has no public unchecked constructor. It exposes the dimensions, compatibility fingerprint, manifest fingerprint, and canonical `dq_dR_relaxed` Jacobian semantics, and public consumers revalidate its sealed canonical manifest. Readers encode both the compatibility fingerprint and per-frame sample ID as `(frame, 32)` `torch.uint8` markers, and grouped readers require one compatibility fingerprint while retaining the validated contract registry for all systems.

## 5. RHF direct-response producer

`deepks.deephf.generate_rhf_force_frame(reference, projector_basis=..., e_target=..., f_target=..., response_options=...)` returns an immutable in-memory `RHFForceFrame`. `deepks.deephf.write_rhf_force_dataset(directory, references, projector_basis=..., e_target=..., f_target=..., response_options=...)` generates one or more frames and passes the completed arrays and provenance to the internal strict schema writer.

The producer validates the native RHF reference through the P2 capability layer, snapshots PySCF-private basis and ECP metadata through the isolated 2.14 compatibility adapter, constructs DeePHF with a zero correction model to obtain a model-independent descriptor Jacobian, validates descriptor differentiability with recorded numerical controls, runs the direct RHF response, checks the independent response residual and invariants, and verifies `dq_dR_relaxed = dq_dR_explicit + dq_dR_response` before forming the persistent arrays.

The base force is the negative native RHF gradient. Target correction labels are formed from the supplied total labels, and all arrays remain `float64` in atomic units. A multi-frame dataset requires the same atom ordering, AO basis, projector, SCF-control signature, occupations, and tensor dimensions for every frame.

All frames and direct responses are completed in memory before the output writer is called. A reference, descriptor, target, response, or cross-frame validation failure therefore produces no dataset directory, and a writer failure removes files created by that write attempt before propagating the error.

The producer support domain is the P2 exact native `pyscf.scf.hf.RHF` domain: converged closed-shell molecular RHF with real spherical AOs, a stable occupied-virtual response, accepted PySCF 2.14 adapter semantics, an all-electron point-nucleus molecule, a fixed compatible projector, finite double-precision state, and a differentiable ordered descriptor. The persisted schema additionally requires a descriptor atom for every raw atom and a descriptor validation result without structural zero blocks.

## 6. Reader, training, and validation

Force-aware data use `force_mode: deephf_relaxed`. In this mode `Reader` loads the strict manifest and exposes `energy`, `descriptor`, `force`, `dq_dR_relaxed`, `force_contract_fingerprint`, and `force_sample_fingerprint`; the corresponding canonical disk fields are `e_corr_target`, `descriptor`, `f_corr_target`, and `dq_dR_relaxed`.

`GroupReader` validates each dataset independently, requires identical exposed fields and descriptor feature counts, and requires one compatibility fingerprint for the group. It retains every validated `ForceDataContract` so each sample ID is checked against the contract that produced that system.

`train_args.force_factor > 0` activates strict relaxed-force training and requires validated contracts. `Evaluator` requires the target energy, descriptor, target force, complete relaxed Jacobian, compatibility marker, and sample marker; it recomputes the runtime frame hashes and checks them against the originating contract before using `predict_correction` for training or validation. It evaluates `energy_factor * energy_loss + force_factor * force_loss` together with the configured auxiliary terms.

Training evaluates the force contraction with `create_graph=True`, so force-loss gradients contain the mixed model derivatives needed to optimize all trainable model branches. Validation uses the same prediction formula without retaining the parameter-gradient graph and reports separate unweighted energy and force MAE/RMSE values for both training and validation datasets.

The projector basis stored in the force contract supplies the model projector when a new force model does not declare one. A declared model is checked for the contract feature count and normalized projector metadata before force training begins.

Energy-only datasets remain valid with `force_mode: none` and `force_factor: 0`. A directory carrying the strict force manifest and markers is accepted only by the relaxed-force reader, evaluator, training, and saved-data paths.

## 7. Checkpoint reload and saved-data testing

Every force-training checkpoint stores `force_training` metadata containing the schema identity and version, compatibility fingerprint, canonical Jacobian semantics, feature count, descriptor definition and spin semantics, shell sizes, projector digest, reference family, and response backend.

`CorrNet.load` requires this metadata in a strict force workflow, checks its canonical values and internal consistency, compares it with the expected dataset contract, and preflights every state entry for exact key, shape, dtype, and finite content before strict loading. It also checks the loaded model input dimension, projector content, shell sizes, and absence of an external element table. Restarted force training repeats these checks before preprocessing or optimization, and resaved checkpoints retain the validated metadata.

`deepks test` accepts `--force-mode deephf_relaxed` or reads the mode from a force-training YAML file. Saved-data testing loads the strict dataset and compatible checkpoint, uses the same correction-energy and relaxed-force predictor as training, reports aggregate and per-system energy and force MAE/RMSE, and writes force rows to `*.force.out` when an output prefix is enabled.

## 8. Explicit failure behavior

| Boundary | Explicit behavior |
|---|---|
| Dataset arrays or manifest | `ForceDataError` reports missing or unexpected fields, dtype, shape, axis, unit, sign, identity, mapping, provenance, convergence, residual, hash, JSON, or overwrite violations. |
| RHF producer | `DeePHFCapabilityError`, `DescriptorDifferentiabilityError`, `RHFResponseError`, or `RHFForceDataError` propagates the unsupported reference, nondifferentiable descriptor, response failure, target error, or cross-frame incompatibility before persistence. |
| Reader and grouping | `ForceDataError` rejects a missing strict manifest, alternate force/Jacobian field names, an incompatible grouped contract, or use of force field names outside strict relaxed mode. |
| Force evaluation and training | `ForceTrainingError`, `TypeError`, or `ValueError` rejects an unsealed contract, missing target, relaxed Jacobian, compatibility marker, sample marker, field hash mismatch, wrong rank, wrong dtype, nonfinite tensor, incompatible model, or invalid force-mode/loss configuration. |
| Checkpoint and restart | Force metadata, compatibility fingerprint, feature count, projector and shell metadata, element-table policy, and every state key, shape, dtype, and finite value are validated before use; an incompatible or incomplete checkpoint raises an error. |
| Saved-data testing | A strict force dataset requires force-aware evaluation and a contract-compatible force checkpoint; an incomplete dataset or an energy-only checkpoint raises an error. |
| Dataset transformation | Legacy statistics helpers reject strict force datasets before rewriting arrays because the operation would invalidate the manifest and hashes. |

No producer, reader, evaluator, training, validation, checkpoint, or saved-data path substitutes `dq_dR_explicit` for a missing or failed `dq_dR_relaxed`.

## 9. Runnable example

`examples/deephf/generate_rhf_force_data.py` builds deterministic distorted-water RHF references, evaluates teacher total energies and forces, and writes strict training and validation datasets. `examples/deephf/rhf_force_train.yaml` trains a compact energy-and-force model on those datasets, and `examples/train_input/force.yaml` provides the general strict force-training configuration form.

Run the vertical slice from the repository root:

```bash
uv run python examples/deephf/generate_rhf_force_data.py
uv run deepks train examples/deephf/rhf_force_train.yaml
uv run deepks test examples/deephf/rhf_force_train.yaml
```

The example selects `force_mode: deephf_relaxed`, `force_name: f_corr_target`, `jacobian_name: dq_dR_relaxed`, a fixed projector basis, deterministic CPU execution, and separate training and validation directories with the same compatibility fingerprint.

## 10. Acceptance coverage and commands

The deterministic P3A suite covers sealed schema round trips and tamper detection, scientific-provenance and convergence failures, projector and shell consistency, response-control and diagnostic consistency, distinct explicit/response/relaxed field export, direct RHF dataset generation, descriptor and correction-force finite differences, multi-frame persistence, response failure without partial output, strict multi-system reader and sample-contract registries, runtime field-hash checks, exact force contraction, force-loss parameter finite differences in linear and dense-network parameters, separate training and validation metrics, end-to-end training initialization and restart, multi-system saved-data metrics, strict checkpoint reload, projector mismatch, state dtype and finite-value checks, and energy-only regression.

Run the acceptance sequence from the repository root:

```bash
uv sync --locked --python 3.11
uv run pytest tests/force_training
uv run pytest tests/analytic_forces
uv run pytest tests/baseline
uv run pytest
uv build
git diff --check
```
