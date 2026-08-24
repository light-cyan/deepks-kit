# Current Project Audit 2

## Review status

**Decision: changes requested; the current revision is not approved.**

The revision contains material improvements, but two reproducible scientific-correctness failures remain: the RKS/UKS validation cache does not cover every state accepted by the full audit, and the force-model contract accepts energy functions whose descriptor dependence is invisible to autograd. The remaining response telemetry, repeated state hashing, derivative recomputation, and retained derived arrays also leave substantial unnecessary computation and memory duplication.

## Scope

This audit covers scientific-correctness risk, redundant computation, and duplicate memory retention in the current production paths for DeePHF gradients, matrix-free response and adjoint calculations, DFT reference validation, force-data reading, training, and evaluation.

## Verification basis

- Full suite: `uv run pytest -q` completed with `827 passed in 96.57s`.
- Detached-dependence reproducer: a hook-free, evaluation-mode `torch.nn.Module` returned energy `2.0`, the production autograd path returned force `-0.0`, and central finite differences returned force `-1.0000000000287557`.
- DFT-cache reproducer: after a strict RKS reference was accepted and cached, assigning an active `with_df` decoration was accepted by `validate_rks_reference`, while `audit_rks_reference` rejected the same object with `the RKS reference has unsupported decorations: density fitting`.
- DFT-hashing probe: one minimal two-atom, 2,000-grid-point RKS direct gradient invoked `_dft_reference_validation_fingerprint` 34 times.
- Static review covered all modified production modules and their modified tests in the current worktree.

## Verified improvements

- Atom selections now reject empty, duplicate, Boolean, negative, and out-of-range indices before gradient execution in `deepks/deephf/gradient.py:17-42`.
- The optimized direct-gradient implementations reuse one descriptor tensor, one model sensitivity, one response solve, and one `dq/dP` construction in their central kernels; the RHF structural budget test confirms one model forward for that path.
- GMRES iterative actions no longer copy and hash every full Krylov vector in `deepks/deephf/adjoint.py:260-270`.
- Strict force readers share NumPy storage with Torch tensors and release the temporary force-array mapping in `deepks/model/reader.py:191-232`.
- Evaluation metrics are aggregated online, and energy-only evaluation runs under `torch.no_grad()` in `deepks/model/train.py:238-274`.
- Training scalar controls are validated before model device or mode mutation in `deepks/model/train.py:562-610`.
- Full RKS and UKS audits remain explicitly callable through `audit_rks_reference` and `audit_uks_reference`.

## Findings

### P0-1: The DFT validation cache key omits audited scientific state

`_audit_rks_reference` rejects active reference decorations including `with_df`, `with_solvent`, `with_x2c`, `mm_mol`, `disp`, and `penalties` in `deepks/deephf/pyscf_rks.py:1007-1024`, while `_dft_reference_validation_fingerprint` does not include those values in `deepks/deephf/pyscf_rks.py:1237-1269`. The same fingerprint also records only the NumInt type, although the full audit validates NumInt backend identity, LibXC version, `omega`, `cutoff`, and instance hooks in `deepks/deephf/pyscf_rks.py:396-431`.

The cache therefore equates states that the authoritative audit treats as scientifically different. The reproduced sequence `validate -> set with_df -> validate` returned the decorated reference without an exception, while the explicit audit rejected it. `RKSDeePHF._reference_state_fingerprint` and `UKSDeePHF._reference_state_fingerprint` reuse this incomplete key, so the method-level scientific-state guard also misses the mutation.

This affects both RKS and UKS because `validate_uks_reference` uses `_dft_reference_validation_fingerprint` in `deepks/deephf/pyscf_uks.py:505-519`.

Required correction: define one canonical state identity that covers every mutable value whose validity is inspected by the corresponding full audit, including decorations and complete NumInt semantics, and add mutation-after-cache tests for every audited field. Cached validation and explicit audit must make the same accept/reject decision for a given state.

### P0-2: The force-model contract permits silently incorrect descriptor derivatives

`validate_force_model` currently establishes evaluation mode and hook absence but accepts arbitrary `torch.nn.Module.forward` implementations in `deepks/deephf/capabilities.py:307-369`. `DeePHF._correction_sensitivity` and `predict_correction` then treat autograd as the complete derivative oracle in `deepks/deephf/method.py:302-341` and `deepks/model/evaluate.py:118-160`.

A model can depend on `values.detach()` while retaining a zero-valued differentiable branch. It passes the current type, dtype, shape, finiteness, mode, and hook checks; its energy changes with the descriptor, but autograd returns zero sensitivity. The production force contraction consequently returns a finite but scientifically wrong force without raising an error. Stochastic and grad-mode-dependent forward implementations are also admitted by the same open model contract.

This is not an argument for restoring finite differences on every gradient or batch. The force-capable model domain must instead be made enforceable: either restrict inference and training to a known differentiable model family and supported operations, or require a derivative audit at model/checkpoint admission and bind its result to an immutable model identity. Regression tests must cover detached descriptor dependence in both DeePHF gradient evaluation and force-aware training.

### P1-3: Matrix-free operator diagnostics still claim measurements and controls that are not executed

`symmetric_operator_telemetry` requires `is_self_adjoint is True` and then assigns `symmetry_residual = 0.0` without evaluating a symmetry residual in `deepks/deephf/adjoint.py:273-305`. Production adapters compare this constant to `operator_symmetry_tolerance`, and response diagnostics persist it as `operator_symmetry_residual`; the value is therefore presented as a measurement even though it is only a backend assertion.

The 16-step Lanczos values are correctly marked as estimates, but response and adjoint dataclasses still expose them as `operator_minimum_eigenvalue`, `operator_maximum_eigenvalue`, and `operator_condition_number` alongside `operator_stability_tolerance`, `operator_condition_tolerance`, and `operator_symmetry_tolerance`, for example in `deepks/deephf/pyscf_rhf.py:340-369`. The production matrix-free path does not apply the stability or condition thresholds to these estimates. The exact bounded-size audit applies the controls, but only when `validate_response_operator_exact` is called explicitly.

The current force schema compounds the semantic mismatch: it requires estimated diagnostics, checks the hard-coded symmetry residual against a tolerance, and accepts any finite condition number greater than or equal to one in `deepks/data/force_schema.py:980-1022`. The stored condition value is no longer cross-checkable because the schema does not retain the minimum absolute Ritz value used by the calculation.

Required correction: represent the self-adjoint property as a contract field rather than a zero residual; name sampled values as Ritz estimates; remove unenforced stability, condition, and symmetry controls from production diagnostics and force-data controls; and expose exact-audit results as a separate certificate with exact eigenvalue semantics.

### P1-4: Non-authoritative operator telemetry adds up to 16 unused operator actions per solve

Every production response or adjoint obtains Lanczos telemetry through `symmetric_operator_telemetry`, which performs up to 16 operator actions in `deepks/deephf/adjoint.py:307-329`. These actions still use `_isolated_problem_action` and an additional `.copy()` in `deepks/deephf/adjoint.py:293-300`, so each diagnostic action retains the copying and SHA-256 work removed from the GMRES inner loop.

The estimates do not gate stability or conditioning, and the symmetry value is constant. RKS additionally runs two independent induced-potential reconstruction probes in `deepks/deephf/pyscf_rks.py:2114-2162`. This diagnostic workload can approach the cost of a small iterative solve while contributing no production acceptance decision.

Required correction: move this telemetry to the explicit audit path or derive diagnostics from vectors already generated by the numerical solver. A production diagnostic must either affect a documented decision or be optional.

### P1-5: DFT cache lookup repeatedly hashes the complete grid and orbital state

The DFT fingerprint scans MO arrays, molecular internals, grid coordinates, grid weights, atom indices, quadrature weights, and the nonzero table in `deepks/deephf/pyscf_rks.py:1237-1269`. Method-level `_assert_science_state`, cached reference validation, response adapters, native gradients, and assembly boundaries repeatedly invoke this full fingerprint.

The measured minimal RKS direct gradient called the full DFT fingerprint 34 times for a 2,000-point grid. The cache avoids complete grid reconstruction, finite differences, and dense quadrature after the first validation, but its lookup remains `O(grid size)` and is repeated at fine-grained internal boundaries.

Required correction: establish one validation transaction per public energy, response, adjoint, or gradient call; compute the complete state identity once at entry and once at the final trust boundary; pass the validated token through internal helpers; and avoid rehashing immutable inputs between hook-free internal operations.

### P1-6: Public derivative paths still repeat model, descriptor, and coordinate derivatives

`DeePHF.dq_dR_relaxed()` calls `response()`, which evaluates force compatibility, and then calls `dq_dR_response()`, which evaluates force compatibility again in `deepks/deephf/method.py:537-567`. The RKS and UHF variants retain the same pattern.

`UHFDeePHF.dq_dR_explicit_spin()` evaluates two spin-component coordinate derivatives and then recomputes the full explicit derivative solely for an internal equality audit in `deepks/deephf/uhf_method.py:91-114`. `UHFDeePHF.dq_dR_relaxed()` recomputes both `dq_dR_explicit()` and `dq_dR_explicit_spin()` again in `deepks/deephf/uhf_method.py:209-241`. These coordinate derivatives are substantially more expensive than summing already available components.

RKS, UHF, and UKS direct drivers validate `atmlst` but call a full-system `_kernel()` and slice only the final result, as shown in `deepks/deephf/rks_gradient.py:245-254` and `deepks/deephf/uhf_gradient.py:215-224`. Their response, native-gradient, explicit-derivative, and retained intermediate arrays therefore scale with all atoms even when a subset is requested. The Z-vector drivers follow the same final-slice pattern.

Required correction: introduce one internal force-input transaction carrying descriptor values, sensitivity, diagnostics, and `dq/dP`; derive spin totals directly from computed components; and propagate validated atom indices into native, explicit, response, and assembly operations. Add call-budget tests for public relaxed-derivative APIs and all four reference families, not only the RHF direct gradient.

### P2-7: Response and gradient objects retain multiple full derived representations

`RHFResponse` stores total, occupied-virtual, and metric components independently for MO response, coefficient response, and density response in `deepks/deephf/pyscf_rhf.py:372-392`. Each field is converted to its own immutable byte-backed array during response construction. The total is the sum of the two components, coefficient response is derivable from MO response, and density response is derivable from coefficient response, so the object retains several complete coordinate-dependent representations of the same solution.

UHF and UKS multiply this pattern across alpha, beta, and spin-summed arrays. Direct gradient drivers then retain the full response object together with explicit, response, and relaxed descriptor Jacobians, spin-resolved and spin-summed copies, and several additive correction-gradient partitions; `deepks/deephf/uhf_gradient.py:185-214` shows the retained result set.

Required correction: choose one canonical stored response representation, expose inexpensive derived quantities through on-demand properties, and provide a compact production result mode that retains only the final gradient, required diagnostics, and the force-data Jacobian when requested. Peak-memory tests should use storage identity and measured retained bytes rather than field-count assertions.

### P2-8: Force batches carry repeated contract markers without preserving frame identity

The reader repeats the same 32-byte compatibility fingerprint for every frame and includes it in every sampled tensor batch in `deepks/model/reader.py:212-241`. The evaluator constructs and compares an expanded expected marker on every batch in `deepks/model/train.py:447-456`. This is duplicate metadata and repeated validation work.

At the same time, the marker proves only compatibility; it does not bind the supplied energy, descriptor, force, and Jacobian tensors to a manifest frame. A caller can combine arbitrary same-shaped tensors with a copied compatibility marker, and the public evaluator accepts them as a strict force sample. The expensive per-frame array hashing need not return, but the trust boundary must be explicit.

Required correction: keep contract identity once on a validated reader or dataset handle, issue batches from that trusted handle, and pass the handle identity separately from tensor payloads. Remove the per-frame repeated marker after the evaluator can distinguish trusted reader batches from arbitrary mappings.

## Approval conditions

Approval requires correction of P0-1 and P0-2 with regression tests, truthful operator-diagnostic semantics for P1-3, and removal or reuse of the production telemetry work in P1-4. P1-5 and P1-6 require call-budget evidence showing that DFT state validation and public derivative calculations no longer repeat full scientific-state scans or coordinate derivatives within one transaction. P2-7 and P2-8 require a documented canonical ownership model and retained-memory tests.

## Final assessment

The current revision is directionally better on the central direct-gradient kernel, GMRES inner iterations, reader storage, and evaluation aggregation. It is not ready for approval because the cache can silently accept a scientifically unsupported DFT state and the open force-model contract can silently produce a wrong force. The remaining telemetry, state hashing, derivative audits, and retained derived arrays also prevent the revision from meeting its stated computational-efficiency and memory goals.
