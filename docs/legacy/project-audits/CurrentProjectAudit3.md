# Current Project Audit 3

## Review status

**Decision: changes requested; revision `feb76230e107833471f68bf2dd6a11a68071994f` is not approved.**

The current revision establishes useful transaction, graph, response-ownership, and batch-metadata structures, but three scientific-correctness blockers remain. The DFT cache can still accept a NumInt implementation that the authoritative audit rejects, the force graph can still be replaced below the checked CorrNet and DenseNet roots, and an accepted ReLU CorrNet can reach a nondifferentiable energy point. Matrix-free adjoint checks, on-demand density reconstruction, atom-subset drivers, retained gradient results, and per-batch model validation also retain material unnecessary computation or duplicate memory.

## Scope

This audit is limited to scientific-correctness risk, redundant computation, and duplicate memory retention in the current DeePHF force, DFT response, adjoint, reader, training, and evaluation paths.

## Verification basis

- Reviewed revision: `feb76230e107833471f68bf2dd6a11a68071994f` (`Reduce force response redundancy and harden science guards`).
- Full suite: `uv run pytest -q` completed with `809 passed in 86.68s`.
- Objective suite: `uv run pytest -q tests/rks_analytic_forces` completed with `139 passed in 10.92s`.
- Independent test: `uv run pytest -q tests/rks_analytic_forces/test_rks_strict_contract.py::test_rks_rejects_a_noncanonical_libxc_parameter_signature` failed because the expected `DeePHFCapabilityError` was not raised.
- DFT-cache reproducer: after one successful validation, replacing `dft.numint.NumInt.eval_xc_eff` caused cached validation to return `ACCEPTED`, while `audit_rks_reference` rejected the same reference with a canonical-parameter residual of `1.000e-07`.
- Leaf-graph reproducer: an exact, hook-free, evaluation-mode CorrNet with only `model.linear.forward` replaced passed `validate_force_model_architecture`; its energy was `1.9999999800000003`, its production autograd force was `-0.0`, and its central-finite-difference force was `-0.9999999900367484`.
- ReLU reproducer: an unmodified supported CorrNet graph at a hidden preactivation of exactly zero passed validation; its production autograd force was `-0.0`, while its central-finite-difference force was `-0.4999999950000001`.
- Adjoint action probe: a two-dimensional symmetric problem required eight transpose actions and one forward action; four transpose actions came from the two two-probe fingerprints, and the three retained residual arrays were equal but did not share storage.
- Response-allocation probe: two reads of `RHFResponse.density_response` were numerically equal but did not share storage, and the complete density equaled the sum of the separately reconstructed metric and occupied-virtual densities.
- Static review covered the modified production modules and their current regression tests; private working files and legacy documentation were not used.

## Verified current properties

- RKS and UKS validation fingerprints cover reference decorations, orbital and molecular state, NumInt instance state, grid configuration, complete grid arrays, and selected global grid-response function identities in `deepks/deephf/pyscf_rks.py:1219-1269`.
- Public RKS and UKS gradient transactions have call-budget tests requiring two complete DFT fingerprints and one force-input evaluation per calculation.
- Force derivatives require an exact CorrNet, exact supported container types, checked root forward implementations, and empty local and global hook registries in `deepks/model/model.py:687-773`.
- Production response objects retain canonical MO responses and expose coefficient and density forms through derived properties, as shown for RHF in `deepks/deephf/pyscf_rhf.py:369-441`.
- Production response and adjoint diagnostics expose `operator_is_self_adjoint` as a Boolean contract field.
- ForceBatch carries contract object references per batch and does not place contract or sample fingerprint tensors on every frame in `deepks/model/reader.py:17-43`.

## Findings

### P0-1: The DFT cache still omits audited NumInt implementation identity

The authoritative functional audit executes `integration.eval_xc_eff` and compares its values with the canonical LibXC calculation in `deepks/deephf/pyscf_rks.py:427-460`. The cache fingerprint records the exact NumInt type, backend identity and version, scalar controls, custom-functional membership, and names of callable instance attributes, but it does not record the identity or implementation of `NumInt.eval_xc_eff` or the other class-level NumInt methods used by production calculations in `deepks/deephf/pyscf_rks.py:1239-1245`. A cache hit therefore bypasses an audited behavior change at `deepks/deephf/pyscf_rks.py:1282-1283`.

The reproduced sequence `validate -> replace NumInt.eval_xc_eff -> validate` was accepted, while the explicit audit rejected the same state. UKS uses the same incomplete fingerprint at `deepks/deephf/pyscf_uks.py:511-524`, so the defect affects both DFT families.

The existing regression is order-dependent. It passes as part of the RKS directory because earlier functional-alias tests leave the cached fingerprint different from the restored reference state, forcing a full audit; it fails when selected independently because the canonical cache entry remains valid. This also means the full-suite pass does not establish the intended mutation-after-cache behavior.

Required correction: bind every class-level or global callable whose semantics are trusted by the full DFT audit and production response path into the canonical scientific identity, or capture and require exact supported implementations before caching. Add a self-contained `validate -> mutate -> validate` regression for RKS and UKS and require it to pass both independently and in the full suite.

### P0-2: Leaf module forward replacement bypasses the force graph contract

`validate_force_model_architecture` rejects instance replacements on the CorrNet root, DenseNet root, and supported embedders, but it only checks that the correction linear layer and DenseNet layers have exact `nn.Linear` types in `deepks/model/model.py:720-731`. It does not reject a `forward` entry in an individual linear module's instance dictionary and does not bind `nn.Linear.forward` itself.

The reproduced exact CorrNet used a leaf `linear.forward` that evaluated `values.detach()` while preserving a zero-valued differentiable branch. The graph passed validation and produced a finite zero force even though finite differences gave a force near negative one. The same validator is used by inference and force-aware training at `deepks/model/evaluate.py:89-124`, so both paths are affected.

Required correction: validate every executable leaf and framework callable in the force graph, including per-instance overrides, or reconstruct a private force model from validated CorrNet metadata and tensors so externally replaceable module execution is not used. Add inference and force-training regressions for the correction linear layer and every DenseNet linear layer.

### P0-3: The supported force model domain contains a nondifferentiable activation

The force activation allowlist includes `torch.relu` in `deepks/model/model.py:691-700`. ReLU is not differentiable at zero, so an exact built-in CorrNet can satisfy every graph and state check while failing the analytic-force premise at an ordinary zero preactivation. The reproducer returned autograd's selected subgradient of zero while symmetric finite differences returned approximately negative one half.

This is distinct from graph replacement: the unsupported scientific state is produced by the current built-in graph itself. Zero biases and symmetric inputs make the kink reachable without malformed parameters.

Required correction: restrict force-capable checkpoints to continuously differentiable activations, or define and enforce a scientific contract that excludes every activation kink for every evaluated frame. A strict analytic-force path must not silently present one framework subgradient as a unique nuclear derivative where the correction energy is nondifferentiable.

### P1-4: ForceBatch identity is forgeable and does not bind tensors to validated frames

`ForceBatch` is a public mutable dictionary with an unrestricted public constructor and a freely assignable `force_contracts` attribute in `deepks/model/reader.py:17-22`. `Evaluator.evaluate` accepts a batch solely when its exact type carries contract objects whose identities occur in the evaluator registry in `deepks/model/train.py:416-426`. Any caller holding an accepted contract can construct `ForceBatch(arbitrary_values, (contract,))`, and a genuine reader batch can be modified after issue without invalidating that identity.

The current tests explicitly construct ForceBatch objects outside Reader, confirming that reader issuance is not enforced. Shape, dtype, and output-finiteness checks remain useful, but they do not establish that energy, descriptor, force, and relaxed Jacobian belong to one validated manifest frame.

Required correction: make batch issuance a non-public operation bound to a validated reader or dataset handle, bind batch membership to immutable frame selections, and keep contract identity outside the tensor payload. The design can avoid repeated full-array hashing, but the evaluator must not describe publicly constructible and mutable mappings as reader-origin evidence.

### P1-5: Matrix-free adjoint production paths still execute redundant operator probes and retain duplicate residual vectors

Every GMRES solve computes a two-action operator fingerprint before the solve and repeats the same two actions afterward in `deepks/deephf/adjoint.py:262-304` and `deepks/deephf/adjoint.py:477-484`. After GMRES, the solver also evaluates both `apply_transpose(solution)` and `apply(solution)` in `deepks/deephf/adjoint.py:449-462`. The RHF and UHF production problems implement `apply_transpose` by calling `apply`, and the RKS transpose path also delegates to `apply`, so the two residual actions evaluate the same self-adjoint operator.

For matrix-free GMRES, `solver_residual` and `transpose_residual` are constructed from the same `transpose_image`, while `physical_residual` is the same mathematical residual for the accepted self-adjoint production problems. All three are converted to independent immutable byte-backed arrays in `deepks/deephf/adjoint.py:485-505` and retained in `AdjointResult`. The two-dimensional probe measured eight transpose actions and one forward action, with three equal non-sharing residual arrays.

Required correction: use the supplied scientific-state fingerprint for entry and exit identity without action probes on exact internal operators, evaluate one independent residual for the self-adjoint production contract, and retain one residual vector plus named scalar diagnostics. Keep distinct forward and transpose audits only in the explicit operator-audit path or for a genuinely nonsymmetric protocol.

### P1-6: On-demand response properties repeat AO transformations already performed by the response solve

The RHF response solve constructs metric, occupied-virtual, and complete AO density responses for invariants in `deepks/deephf/pyscf_rhf.py:2140-2181`, but the returned object retains only the MO response. Each density property later rebuilds an MO partition, coefficient response, AO density, symmetrized result, and immutable byte copy in `deepks/deephf/pyscf_rhf.py:385-441`.

Direct gradients immediately request the complete, metric, and occupied-virtual properties separately, for example in `deepks/deephf/rks_gradient.py:150-174`. UHF and UKS additionally allocate three spin stacks from six independently reconstructed spin densities in `deepks/deephf/uhf_gradient.py:93-138` and `deepks/deephf/uks_gradient.py:59-76`. Repeated property reads repeat the entire transformation and allocate new storage every time.

Required correction: provide one internal density-partition bundle derived in a single pass and consumed once by gradient assembly, or pass the transient density bundle directly from the solve without retaining it in the public response. Preserve canonical long-lived ownership while eliminating repeated transformations and byte copies in the same calculation.

### P1-7: Most atom-subset gradient drivers still compute the full molecule and slice only the result

RKS direct validation obtains `atom_indices`, calls an argument-free full `_kernel`, and slices `de_full` afterward in `deepks/deephf/rks_gradient.py:243-256`. UHF direct does the same in `deepks/deephf/uhf_gradient.py:213-226`, and UKS inherits that kernel. RKS and UHF Z-vector drivers use the same final-slice structure in `deepks/deephf/rks_zvector.py:328-341` and `deepks/deephf/uhf_zvector.py:326-339`; UKS inherits the UHF implementation. RHF Z-vector likewise computes native and explicit full-system derivatives before selection in `deepks/deephf/zvector.py:78-118`.

The RHF direct driver already propagates selected atoms into native, explicit, and response calculations, demonstrating the expected bounded path. The remaining drivers still allocate and contract full coordinate-axis arrays for a one-atom request.

Required correction: propagate validated atom indices through native gradients, explicit descriptor derivatives, direct responses, adjoint nuclear contractions, and result assembly for every family and backend. Add call and shape budgets proving that a selected calculation never constructs a full coordinate-dependent Jacobian or gradient partition.

### P2-8: Gradient drivers retain additive and spin-summed copies of large results

RKS direct retains explicit, response, and relaxed descriptor Jacobians together even though the relaxed array is their sum, and also retains every additive correction partition in `deepks/deephf/rks_gradient.py:213-240`. UHF and UKS retain the same three Jacobians in spin-resolved and spin-summed forms plus spin-resolved and summed correction partitions, as shown in `deepks/deephf/uhf_gradient.py:186-210` and `deepks/deephf/uks_gradient.py:93-118`. Attaching the canonical response or adjoint object to the driver adds its coordinate-dependent state to the same long-lived result.

These retained arrays are useful for an explicit scientific audit or force-data export, but ordinary gradient and force calls require the final gradient and compact diagnostics. Canonical response storage alone therefore does not resolve production result retention.

Required correction: add a compact production result mode that retains the final gradient and required diagnostics, and materialize descriptor Jacobians or scientific partitions only for an explicit audit or force-data request. Verify retained bytes and storage identities for RHF, RKS, UHF, and UKS direct and Z-vector drivers.

### P2-9: Force evaluation repeats whole-model and contract validation for every batch

Every force batch re-runs `validate_model_force_contract` in `deepks/model/train.py:407-408`, graph and hook traversal in `deepks/model/evaluate.py:89-91`, and a finiteness scan over every floating parameter and buffer in `deepks/model/evaluate.py:39-55`. Contract validation reparses immutable manifest data, while graph structure and contract compatibility do not change between batches in one training or evaluation transaction. On accelerators, `torch.isfinite(...).all()` over every model tensor also introduces repeated device reductions and synchronization risk.

Required correction: validate immutable graph and contract state once when the evaluator binds a model and reader registry, validate parameter values once per optimizer-update boundary or checkpoint admission, and keep only batch-dependent tensor checks in the batch hot path. Bind the cached validation to module structure and parameter version evidence so an actual mutation invalidates it.

## Approval conditions

Approval requires all three P0 findings to be corrected with independent regressions. Cached RKS and UKS validation must reject class-level implementation changes after a cache hit; force inference and force-aware training must reject every leaf execution override; and the force-capable activation domain must guarantee differentiability.

Approval also requires a trustworthy ForceBatch issuance boundary, one-action self-adjoint residual verification without duplicate retained vectors, one-pass density partition derivation, and atom-subset call budgets for every current family and backend. Compact gradient ownership and transaction-level model validation require retained-byte and call-count evidence rather than field-count assertions.

## Final assessment

The current revision is materially closer to the intended design: DFT validation uses a coarse transaction boundary, production diagnostics use explicit self-adjoint contracts, response objects have one canonical orbital-response representation, and force batches no longer carry repeated marker tensors. It is still not scientifically safe to approve because cached DFT semantics and the enforced force graph both remain bypassable, while the built-in ReLU domain admits a nonanalytic force. The remaining operator probes, density reconstructions, full-system subset paths, retained derivative partitions, and per-batch validation prevent approval on the requested computation and memory criteria.
