# Strict RHF relaxed-force training

This example generates a deterministic RHF DeePHF dataset governed by the strict force-data contract and trains an energy-correction model against both `e_corr_target` and `f_corr_target`. The force prediction contracts the model's complete descriptor sensitivity with `dq_dR_relaxed`.

Run the example from the repository root:

```bash
uv run python examples/deephf/generate_rhf_force_data.py
uv run deepks train examples/deephf/rhf_force_train.yaml
```

The generator uses native, converged PySCF RHF references and writes one training dataset and one validation dataset. Energies use `Eh`, coordinates use `Bohr`, forces use `Eh/Bohr` with `force=-dE/dR`, and the relaxed descriptor Jacobian uses `Bohr^-1` with `+dq/dR`.

For application data, call `deepks.deephf.write_rhf_force_dataset(directory, references, projector_basis=..., e_target=..., f_target=...)` with float64 target energies and forces. The producer evaluates the complete direct RHF response before the strict schema writer atomically persists the canonical fields and provenance manifest.

The training configuration selects `force_mode: deephf_relaxed`, the canonical `f_corr_target` label, and the canonical `dq_dR_relaxed` Jacobian. The training and validation datasets share the same compatibility fingerprint, including their projector, reference, response, descriptor, axis, sign, and unit semantics.

Strict molecular inference is available through the public CLI:

```bash
uv run deepks deephf examples/deephf/rhf_inference.yaml
```

The inference example constructs a fresh native RHF reference for `H2`, evaluates the complete direct analytic gradient with a zero correction, and writes canonical energy, descriptor, gradient, and force arrays under `examples/deephf/inference_output`.
