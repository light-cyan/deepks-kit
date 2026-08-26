# Strict RHF relaxed-force training

This example generates a deterministic RHF DeePHF dataset governed by the strict force-data contract and trains an energy-correction model against both `e_corr_target` and `f_corr_target`. The force prediction contracts the model's complete descriptor sensitivity with `dq_dR_relaxed`.

Run the example from the repository root:

```bash
srun --gres=gpu:1 uv run python examples/deephf/generate_rhf_force_data.py
srun --gres=gpu:1 uv run deepks train examples/deephf/rhf_force_train.yaml
```

The generator converges each RHF reference with GPU4PySCF and writes one training dataset and one validation dataset. Energies use `Eh`, coordinates use `Bohr`, forces use `Eh/Bohr` with `force=-dE/dR`, and the relaxed descriptor Jacobian uses `Bohr^-1` with `+dq/dR`.

For application data, call `deepks.deephf.write_rhf_force_dataset(directory, references, projector_basis=..., e_target=..., f_target=..., target=...)` with float64 target energies and forces plus the canonical target calculation identity described in `docs/DeePHFDataIntegrityRequirements.md`. The producer evaluates the complete direct RHF response before the strict schema writer atomically persists the canonical fields and provenance manifest.

The training configuration selects `force_mode: deephf_relaxed`, the canonical `f_corr_target` label, and the canonical `dq_dR_relaxed` Jacobian. The training and validation datasets share the same compatibility fingerprint, including their target identity, projector, reference, response, descriptor, axis, sign, and unit semantics.

Strict molecular inference is available through the public CLI:

```bash
srun --gres=gpu:1 uv run deepks deephf examples/deephf/rhf_inference.yaml
```

The inference example converges an RHF reference for `H2` with GPU4PySCF, evaluates the complete direct analytic gradient with a zero correction, and writes canonical energy, descriptor, gradient, force, and electronic-root provenance under `examples/deephf/inference_output`. For multi-frame systems, each accepted density initializes the next frame and the configured occupied-subspace overlap threshold rejects discontinuous roots before output is published.
