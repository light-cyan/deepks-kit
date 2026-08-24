# Current Project Audit 5

## Review status

**Decision: changes requested; revision `1680f7f` is not approved.**

The current revision correctly detects ordinary PyTorch call replacement, common ForceBatch alias mutation, matrix-free false self-adjoint claims, and unavailable selected-coordinate translation diagnostics. It also reduces returned gradient ownership and executes one native DFT gradient. Approval remains blocked because a masking descriptor can still replace the executed CorrNet graph, native RKS and UKS gradient implementations are outside the trusted DFT implementation identity, and the dense adjoint path still permits a false self-adjoint declaration to suppress the physical residual. Compact direct and Z-vector execution also continues to materialize descriptor Jacobians, unused AO density partitions, complete adjoint partition objects, and avoidable hashing copies.

## Scope

This audit is limited to scientific-correctness risk, redundant computation, and duplicate or avoidable memory materialization in the current force-model, ForceBatch, response, adjoint, descriptor derivative, direct-gradient, Z-vector, and DFT native-gradient paths.

## Verification basis

- Reviewed revision: `1680f7f` (`Resolve audit 4 correctness and redundancy`).
- Full suite: `uv run pytest -q` completed with `815 passed in 83.43s`.
- Masked-dispatch reproducer: a descriptor installed as `CorrNet.__call__` returned the trusted `nn.Module.__call__` when accessed through the class and a detached graph when bound to an instance; architecture validation accepted it, the analytic force was `-0.0`, and the central-finite-difference force was `-0.24999999749530932`.
- Dense-adjoint reproducer: a nonsymmetric dense operator declaring `is_self_adjoint = True` was accepted with a stored residual of `0.0` and a physical forward residual of `1.0` under `require_physical_residual=True`.
- Native-RKS reproducer: after successful cached reference validation, replacing `rks_grad.Gradients.kernel` with a finite float64 zero-gradient implementation was accepted by both reference validation and `native_rks_gradient`; the selected gradient returned `[[0.0, 0.0, 0.0]]`.
- Current density call-budget tests establish two post-solve AO transformations for compact restricted direct gradients and four for compact unrestricted direct gradients.
- Static call-path review confirmed that compact explicit-gradient contraction calls the full per-shell Jacobian builder, compact AO-potential construction independently builds complete `dq/dP`, and compact Z-vector drivers call the standard complete adjoint solve.
- Private working files and legacy documentation were not used.

## Verified current properties

- Direct assignment to `_compiled_call_impl` and ordinary class-level replacement of `CorrNet.__call__` are rejected for force inference and force-aware training.
- The accepted activation set excludes ReLU, and root, DenseNet, embedding, linear-forward, hook, `__call__`, `_wrapped_call_impl`, `_call_impl`, and compiled-call evidence is present in the ordinary force-model validation path.
- Reader-issued ForceBatch objects use registered issuer instances and per-frame content fingerprints. Ordinary tracked Torch mutation, NumPy-view mutation, and `.data` mutation change accepted content and are detected at evaluation.
- Matrix-free self-adjoint adjoint solves use the physical forward action for their independently retained final residual.
- Selected RKS, UKS, and unrestricted response diagnostics represent the full-coordinate translation residual as unavailable instead of recording a synthetic zero, and their explicit response audits reproduce that domain.
- Detailed direct response and gradient assembly share transient AO density partitions rather than reconstructing the same partitions after response solution.
- Selected-atom direct and Z-vector gradients agree with their full-coordinate rows in the current numerical tests.
- Compact returned driver state retains the final gradient and diagnostics rather than long-lived coordinate Jacobians and response or adjoint objects.
- Native RKS and UKS reference gradients are evaluated once per production gradient call.

## Findings

### P0-1: A masking descriptor bypasses the force-model call-dispatch contract

`validate_force_model_architecture` compares `nn.Module.__call__`, `_wrapped_call_impl`, and `_call_impl` with captured implementations and then evaluates each executed module's dispatch through `getattr(type(module), name)` in `deepks/model/model.py:719-778`. `force_model_structure_evidence` obtains the same class attributes dynamically in `deepks/model/model.py:808-875`.

Dynamic attribute access executes the descriptor protocol. A descriptor installed on the exact CorrNet class can therefore return the captured trusted function when accessed with `instance is None`, while returning a detached implementation when Python binds the special method to a model instance. Exact model type, ordinary class identity, instance override, compiled-call, and hook checks all remain satisfied because validation observes the descriptor's masked class result rather than its raw definition.

The reproduced model passed `validate_force_model_architecture` and `predict_correction`, returned an analytic force of zero, and retained a finite-difference force near negative one quarter. The same architecture validator and structure evidence are used by the force-aware evaluator, so cached training validation is affected as well as direct inference.

Required correction: inspect special-method definitions without invoking descriptors, for example through `inspect.getattr_static` and explicit MRO ownership checks. Require the raw descriptor and its defining class to match the captured supported implementation for every executed module. Structure evidence must include the raw static descriptor identities and owners. Add masked data-descriptor and non-data-descriptor regressions for direct prediction and cached evaluator execution.

### P0-2: Native DFT gradient implementations are outside the trusted scientific identity

The DFT implementation contract captures selected NumInt and LibXC methods, supported RKS and UKS reference methods, grid builders, and `rks_grad.grids_response_cc` in `deepks/deephf/pyscf_rks.py:45-81`. The validation fingerprint records those identities in `deepks/deephf/pyscf_rks.py:1270-1317`, but it does not record `rks_grad.Gradients.kernel` or the native gradient callables on which UKS assembly depends.

`native_rks_gradient` constructs an exact PySCF RKS gradient driver, enables grid response, and directly accepts the result of `driver.kernel` after shape, dtype, finiteness, and unchanged-reference checks in `deepks/deephf/pyscf_rks.py:3849-3876`. Those checks cannot distinguish a scientifically wrong finite implementation. The reproduced cached reference accepted a replacement kernel that returned a float64 zero gradient of the expected shape.

`native_uks_gradient` similarly trusts the exact UKS gradient driver and `_native_unrestricted_gradient` assembly in `deepks/deephf/pyscf_uks.py:1084-1112`. That assembly executes driver methods including core, overlap, energy-density, effective-derivative, nuclear-gradient, and tagged-density operations whose implementations are not part of the current DFT fingerprint.

The previous independent reconstruction is not required on every production call if implementation identity is complete. Required correction: capture and require the raw identities of every RKS and UKS gradient callable and mutable module-level dependency reached by the native gradient path, and include them in both reference-validation fingerprints and transaction evidence. Alternatively, establish a numerical native-gradient audit certificate once per unchanged implementation and scientific state. Add cached `validate -> replace gradient implementation -> gradient` regressions for RKS and UKS.

### P1-3: Dense adjoint self-adjoint claims still suppress the physical residual

The matrix-free branch selects `problem.apply` for its final residual when `is_self_adjoint` is true in `deepks/deephf/adjoint.py:441-462`. The dense branch instead always computes `final_matrix.T @ solution - objective_gradient` in `deepks/deephf/adjoint.py:434-440`. The subsequent physical audit runs only when `is_self_adjoint` is not exactly true in `deepks/deephf/adjoint.py:481-490`.

A nonsymmetric dense problem can therefore assert self-adjointness, solve the transpose equation exactly, and skip the physical forward equation. The reproduced two-dimensional problem retained a zero transpose residual while its forward residual was one. This contradicts the public `require_physical_residual=True` contract even though the internal production operators currently intend to be symmetric.

Required correction: select the physical forward action for the single final self-adjoint residual in both dense and matrix-free modes. For dense mode, compute `final_matrix @ solution - objective_gradient` when the accepted self-adjoint contract is being verified, or invoke `problem.apply` if the dense matrix itself is not the authoritative physical action. Add a dense false-claim regression alongside the matrix-free regression.

### P1-4: Compact explicit-gradient contraction still materializes descriptor Jacobians

`contract_dq_dR_explicit` states that it contracts fixed-density descriptor motion without materializing `dq/dR`, but it calls `_dq_dR_explicit_shells` in `deepks/descriptor/derivatives.py:144-165`. That helper constructs all projected-density blocks, all full eigenvalue Jacobians, all coordinate projected-density Jacobians, and a list of `bxav` descriptor-coordinate Jacobians in `deepks/descriptor/derivatives.py:86-117`. Avoiding only the final concatenation does not avoid the coordinate Jacobian calculation or its peak shell-array ownership.

After the explicit contraction, compact direct and Z-vector paths call `_correction_ao_potential`, which builds complete `dq/dP` when no array is supplied in `deepks/deephf/method.py:388-418`. `dq/dP` independently reconstructs the same projected-density blocks and eigenvalue Jacobians in `deepks/descriptor/core.py:50-67`. One compact gradient therefore evaluates and materializes the descriptor differential twice even though both consumers use the same model sensitivity.

Required correction: use the sensitivity as a vector-Jacobian seed and compute the projected-density or AO adjoint once. Derive both the AO correction potential and explicit projector-motion gradient from that contracted adjoint without constructing full `dq/dP`, `dq/dR`, per-feature eigenvalue Jacobians, or a list of all shell-coordinate results. Detailed force-data and audit modes may retain the full Jacobians. Add call, shape, and peak-live-byte tests below the public `dq_dR_explicit` method boundary so internal shell materialization cannot satisfy a false compact budget.

### P1-5: Compact direct response computes density partitions that it does not consume

The restricted response solver always constructs metric and occupied-virtual MO arrays, transforms both to AO densities, and materializes their sum in `deepks/deephf/pyscf_rhf.py:2151-2167`. Compact RKS and RHF direct drivers request `result_mode="gradient"` but consume only the complete density at index zero of the returned partition bundle, as shown for RKS in `deepks/deephf/rks_gradient.py:141-160`. Metric and occupied-virtual AO arrays are discarded.

The unrestricted response solver constructs metric and occupied-virtual AO densities for both spins under every non-response result mode in `deepks/deephf/pyscf_uhf.py:1563-1596`. Compact UHF and UKS drivers consume only the two complete spin densities in `deepks/deephf/uhf_gradient.py:161-184` and `deepks/deephf/uks_gradient.py:112-128`. Current tests explicitly require two restricted or four unrestricted transformations, but those are detailed-partition budgets rather than compact-gradient minima.

For compact gradient assembly and current invariants, one complete restricted transformation or one complete transformation per spin is sufficient. Required correction: distinguish a complete-density compact mode from the detailed partition mode, reducing the compact call budget to one restricted or two unrestricted AO transformations. Contract the objective while the transient complete density is available and return only diagnostics and the final response-gradient contraction.

Detailed unrestricted assembly also converts each partition tuple through `np.stack` in `deepks/deephf/uhf_gradient.py:88-90` and `deepks/deephf/uks_gradient.py:62-64` while the original component arrays remain live. Required correction: allocate the desired spin representation once in the response adapter or contract the component tuples directly without retaining equal stacked copies.

### P1-6: Compact Z-vector execution still builds a complete adjoint partition object

Every method `_zvector_inputs` path calls the standard adjoint adapter `solve`, as shown for RKS in `deepks/deephf/rks_method.py:167-183`. The returned immutable adjoint includes the objective, solution, residual or canonical adjoint state, and multiple coordinate-gradient partitions. RKS includes fixed-grid, grid-coordinate, grid-weight, nuclear, metric, occupied-virtual, and complete response arrays in `deepks/deephf/pyscf_rks.py:218-241`; unrestricted and UKS objects additionally retain spin-resolved and summed partitions.

Compact Z-vector drivers then use only the adjoint diagnostics and complete `correction_gradient_response`, as shown in `deepks/deephf/rks_zvector.py:169-185`, before discarding the complete object. Returned driver retention is compact, but adjoint partition computation, immutable conversion, integrity fingerprinting, and peak ownership are unchanged.

Required correction: provide an adjoint gradient result mode that retains the scientifically required solve residual and diagnostics internally but directly returns the complete response-gradient contraction without constructing detailed coordinate partitions or a public immutable adjoint object. Preserve the complete object for explicit adjoint inspection and audit requests. Add partition-construction call budgets and peak-allocation tests for RHF, RKS, UHF, and UKS compact Z-vector calculations.

### P2-7: ForceBatch hashing creates a full temporary copy of every hashed frame

`_tensor_fingerprint` obtains a contiguous NumPy representation and calls `array.tobytes()` before updating SHA-256 in `deepks/model/reader.py:24-30`. `tobytes()` necessarily allocates a complete Python bytes copy. `_force_batch_error` invokes this function for every field and every selected frame in every evaluated force batch in `deepks/model/reader.py:91-110`, including the largest relaxed descriptor Jacobian.

Content scanning is required by the chosen guarantee for detecting mutations that bypass Torch version counters, but the second contiguous bytes copy is not required. If the source frame is already contiguous, `hashlib` can consume a contiguous buffer or byte-cast memoryview directly. If canonicalization first requires a contiguous array, the hash still does not require another `tobytes()` allocation.

Required correction: update the digest from a contiguous buffer view, retain explicit dtype and shape framing, and add a peak-memory regression for a large relaxed-Jacobian frame. Measure the per-epoch CPU and memory-bandwidth cost of per-frame SHA relative to force contraction; if it is material, select a documented integrity boundary or batching scheme that preserves mutation detection without avoidable per-field Python and hashing overhead.

## Approval conditions

Approval requires the two P0 findings to be corrected with independent masked-descriptor and native-gradient implementation-replacement regressions. The executed force graph must be identified through raw static dispatch definitions, and every trusted RKS and UKS native-gradient callable must be included in the scientific transaction identity or covered by a state-bound numerical certificate.

Approval also requires dense and matrix-free physical adjoint residual semantics to agree, compact explicit derivatives to avoid full descriptor Jacobians, compact direct density construction to meet one/two transformation budgets, and compact Z-vector execution to avoid complete adjoint partition construction. Tests must measure internal call counts and peak live storage rather than only returned object fields.

ForceBatch content verification is scientifically effective for the tested mutation mechanisms, but its full-frame bytes copies remain within the requested duplicate-memory scope and should be removed before efficiency approval.

## Final assessment

This revision resolves several important Audit 4 defects, including ordinary module dispatch replacement, common ForceBatch alias mutation, matrix-free physical residual selection, selected-coordinate translation semantics, repeated native DFT evaluation, and long-lived compact driver ownership. It is not ready for approval because the current dispatch comparison can be masked by the descriptor protocol, the newly singular native DFT gradient calculation is not protected by the implementation fingerprint, and dense self-adjoint verification remains bypassable. The compact paths also reduce returned storage more than they reduce calculation or peak memory: full descriptor differentials, unused density partitions, complete adjoint partitions, unrestricted density stacks, and per-frame hashing copies remain active.
