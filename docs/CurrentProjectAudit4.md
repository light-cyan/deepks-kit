# Current Project Audit 4

## Review status

**Decision: changes requested; revision `cda593af5c6b0303347157c49219000f2f2ef8f4` is not approved.**

The current revision corrects the previously identified DFT implementation-cache boundary, rejects ReLU and common leaf-forward replacements, propagates selected atoms through the main direct and Z-vector calculations, and reduces long-lived response and Jacobian retention in ordinary gradient results. Approval remains blocked because the force-model execution contract and ForceBatch provenance contract are still bypassable, the requested physical adjoint residual can be skipped by an unverified self-adjoint claim, selected-coordinate response diagnostics contain a false invariant value, density transformations remain duplicated within one calculation, and compact result handling removes large intermediates only after computing and materializing them.

## Scope

This audit is limited to scientific-correctness risk, redundant computation, and duplicate or avoidable memory materialization in the current DeePHF force-model, response, adjoint, reader, gradient, force, and scanner paths.

## Verification basis

- Reviewed revision: `cda593af5c6b0303347157c49219000f2f2ef8f4` (`Resolve audit correctness and response redundancy`).
- Full suite: `uv run pytest -q` completed with `813 passed in 83.18s`.
- Focused DFT-cache and force-prediction suite completed with `35 passed in 8.68s`.
- Independent RKS selected-coordinate calculation agreed with the corresponding rows of the full calculation to a maximum error of `2.113e-13` for the direct path and `4.441e-16` for the Z-vector path.
- Force-dispatch reproducer: assigning an instance `_compiled_call_impl` that returned a detached correction passed `validate_force_model_architecture`; the production autograd force was `-0.0`, while the central-finite-difference force was `-1.0000000000287557`.
- Class-dispatch reproducer: replacing `CorrNet.__call__` with a detached execution path also passed architecture validation and produced the same zero-versus-negative-one force disagreement.
- ForceBatch provenance reproducer: directly constructed `_ForceBatchIssuer` and `_issue_force_batch` objects with arbitrary tensors passed `_force_batch_error` when supplied with the accepted contract object.
- ForceBatch mutation reproducer: mutations through both `tensor.numpy()[...]` and `tensor.data.fill_(...)` left `tensor._version` unchanged and remained accepted by `_force_batch_error`.
- Adjoint reproducer: a protocol object declaring `is_self_adjoint = True`, with `apply(v) = 2v` and `apply_transpose(v) = v`, was accepted with a stored residual of approximately `2.220e-16` even though its physical residual was approximately `1.0`.
- Selected-response reproducer: a strict one-atom RKS response recorded `translation_residual = 0.0`, but the explicit response-equation audit rejected that response because the recorded value was not reproducible from its selected-coordinate density response.
- Static call-path review covered the current production modules and focused regression tests; private working files and legacy documentation were not used.

## Verified current properties

- RKS and UKS DFT validation now binds the supported NumInt and LibXC implementations before a cached validation result is accepted, and the previously order-dependent isolated mutation regression passes.
- Force-capable CorrNet validation rejects ReLU, checks the built-in root, DenseNet, embedding, and linear `forward` implementations, and rejects active module hooks.
- The principal RHF, RKS, UHF, and UKS direct and Z-vector paths propagate validated atom selections into their lower-level native, explicit, response, and adjoint calculations; the independently checked RKS selected-coordinate values agree with full-coordinate calculation rows.
- Matrix-free self-adjoint solve results retain one named residual vector rather than multiple equal residual fields.
- Response objects retain canonical MO response state and derive AO coefficient and density representations on demand.
- Ordinary gradient, force, and scanner entry points select compact result retention, leaving the final gradient and diagnostics instead of retaining the complete response object and descriptor Jacobians after return.
- Reader-issued force batches bind frame selections and tensor versions without per-frame contract and marker tensors, and ordinary in-place Torch operations that increment `_version` are detected.

## Findings

### P0-1: PyTorch module dispatch can replace the validated force graph

`validate_force_model_architecture` binds `CorrNet.forward`, `DenseNet.forward`, supported embedding forwards, `nn.Linear.forward`, and `nn.Module._call_impl` in `deepks/model/model.py:687-739`. Its structure evidence records the same identities in `deepks/model/model.py:784-835`. PyTorch execution reaches `_call_impl` through `nn.Module.__call__` and `_wrapped_call_impl`, and `_wrapped_call_impl` selects an instance `_compiled_call_impl` when that field is populated. None of these dispatch points is bound by the current contract.

An exact built-in CorrNet can therefore pass validation while executing a detached replacement graph. Both the instance `_compiled_call_impl` reproducer and the class-level `CorrNet.__call__` reproducer produced a finite zero autograd force for an energy whose finite-difference force was approximately negative one. Exact type checks, leaf-forward checks, and empty hook registries do not prevent this execution replacement.

Required correction: bind the complete module call-dispatch chain used by the supported Torch version, require `_compiled_call_impl` to be absent or an exact trusted implementation, and include every dispatch identity and instance override in `force_model_structure_evidence`. Add inference and force-aware training regressions for instance compiled-call replacement and class-level `__call__` or `_wrapped_call_impl` replacement. Any validation bypass flag used after a trusted entry check should be confined to an internal capability rather than accepted as a caller-selectable trust assertion.

### P0-2: ForceBatch provenance is forgeable and its mutation evidence is incomplete

`_ForceBatchIssuer` has an unrestricted constructor in `deepks/model/reader.py:22-26`, and `_issue_force_batch` directly exposes the module token in `deepks/model/reader.py:59-60`. `_force_batch_error` accepts a batch when each issuer's `contract` is identical to one accepted contract and each tensor's current `_version` equals its issuance version in `deepks/model/reader.py:63-73`. The check does not bind issuer identity to the actual configured Reader instances or bind tensor content to the selected manifest frames.

The reproduced arbitrary batch created through `_ForceBatchIssuer(accepted_contract)` and `_issue_force_batch(...)` was accepted. A genuine tensor can also be modified through a NumPy view or `.data` without incrementing `_version`, so the current evidence detects common tracked Torch mutations but does not establish immutable batch content. NumPy/Torch shared storage makes external-storage mutation a normal aliasing case rather than only an adversarial API use.

Required correction: register and validate issuer identities created by the actual configured Reader objects, make the issue capability reachable only through that registry, and bind every selection to the issuing dataset. If arbitrary post-issuance mutation must be detected, perform one content-integrity check per evaluated batch or use storage that cannot be mutated through shared aliases; `_version` alone cannot support the current mutation-detection claim. Add forgery, NumPy-view mutation, `.data` mutation, cross-reader selection, split, and concatenation regressions.

### P1-3: A claimed self-adjoint operator bypasses the requested physical residual

After matrix-free GMRES, `solve_scalar_adjoint` computes its retained residual through `problem.apply_transpose` in `deepks/deephf/adjoint.py:442-465`. The independent physical `problem.apply` check runs only when `require_physical_residual` is true and `problem.is_self_adjoint` is not exactly true in `deepks/deephf/adjoint.py:476-485`. Consequently, the unverified self-adjoint declaration disables the check whose purpose is to validate the physical operator.

The reproduced nonsymmetric protocol declared `is_self_adjoint = True`, solved the transpose equation to machine precision, and was accepted despite a physical residual of approximately one. The current internal RHF, RKS, UHF, and UKS problem implementations may implement equal actions, but the public solver semantics do not enforce that equality and cannot use the declaration itself as its evidence.

Required correction: for the self-adjoint production contract, use `problem.apply` for the single independent post-solve residual so that one action verifies both the physical equation and the accepted solution. Alternatively, restrict the optimized path to exact trusted internal problem types and keep an explicit equality audit at contract admission. Add a regression whose forward and transpose actions differ while `is_self_adjoint` is asserted.

### P1-4: Selected-coordinate responses publish a false translation invariant

RKS response construction records the maximum translational density residual for a full atom set but substitutes `0.0` for every selected subset in `deepks/deephf/pyscf_rks.py:2439-2443`. The explicit audit always recomputes the maximum norm of the sum over the response's coordinate axis in `deepks/deephf/pyscf_rks.py:2839-2844`. UHF and UKS use the same zero-substitution concept for their selected alpha, beta, and total translation diagnostics.

A coordinate subset does not contain the complete translational perturbation, so its coordinate-axis sum is not the full-system translation invariant. Recording zero states that a measurement was performed and passed, while the explicit audit demonstrates that the value is generally not reproducible. A valid response produced by the adapter can therefore fail the adapter's own explicit scientific audit.

Required correction: represent the full-coordinate translation invariant as unavailable for selected responses, with an explicit availability field or optional value, and exclude it from selected-response invariant thresholds. A separate full-coordinate audit certificate may be retained when required. Add selected RKS, UKS, RHF, and UHF response tests that require every recorded diagnostic to be reproducible under its declared domain.

### P1-5: AO density transformations are repeated after the response solve

The RHF response solve constructs metric and occupied-virtual AO densities and their sum for scientific invariants in `deepks/deephf/pyscf_rhf.py:2141-2175`. The returned response retains only its MO response, and `density_partitions` performs the two AO transformations again in `deepks/deephf/pyscf_rhf.py:397-415`. The direct gradient immediately requests this reconstructed bundle in `deepks/deephf/gradient.py:287-308`. The completed RHF response and direct-assembly path therefore performs four post-solve partition transformations rather than one shared pair.

The UHF response solve builds complete alpha and beta densities for invariants in `deepks/deephf/pyscf_uhf.py:1572-1581`, while `UHFResponse.density_partitions` later builds metric and occupied-virtual densities for both spins in `deepks/deephf/pyscf_uhf.py:141-159`. These are six post-solve spin-density transformations. The unrestricted partition method also creates immutable component results, copies them into `np.stack` arrays, copies the stacks into new immutable arrays, and materializes another complete summed stack.

Required correction: carry one transient internal density-partition work result from response solution into direct gradient assembly, or derive all required invariant and gradient contractions in one pass without storing density partitions in the public response object. Avoid re-freezing arrays that are already exclusively owned immutable temporaries. Add transformation call budgets and peak-allocation checks for restricted and unrestricted direct calculations.

### P1-6: Compact result mode discards full intermediates only after materialization

The standard RHF direct gradient constructs the explicit descriptor Jacobian, response descriptor Jacobian, relaxed Jacobian, response object, metric and occupied-virtual corrections, and all additive correction gradients in `deepks/deephf/gradient.py:269-346`. Only after the complete result graph exists does `retain_details = False` call `_compact_driver_results` in `deepks/deephf/gradient.py:350-351`, whose implementation resets the driver and restores the final gradient and diagnostics in `deepks/deephf/gradient.py:74-81`. The other families follow the same compute-then-delete ownership model.

This design reduces long-lived memory after return but does not reduce arithmetic, allocation count, or peak memory. An ordinary gradient requires contractions with descriptor sensitivity and the final correction gradient; it does not require materializing the full relaxed descriptor Jacobian unless force-data export or an explicit derivative audit requests it.

RKS native-gradient construction also evaluates the custom grid-coordinate and grid-weight components, a complete PySCF gradient with grid response, and a second PySCF gradient without grid response before reconstructing their equality in `deepks/deephf/pyscf_rks.py:3920-3979`. UKS has the corresponding structure. Standard compact gradient, force, and scanner calls pay for this full partition audit even though the detailed partitions are discarded.

Required correction: select the compact computational plan before response and native-gradient assembly. Directly contract explicit and response contributions into the final correction gradient, materialize descriptor Jacobians only for force-data or explicit audit requests, and reserve independent DFT grid reconstruction for an explicit full-audit entry point or a state-bound reusable audit certificate. Verification must measure call counts and peak live bytes, not only retained fields after return.

### P2-7: Adjoint immutable-result construction creates avoidable transient copies

The matrix-free solution is converted to an immutable byte-backed array in `deepks/deephf/adjoint.py:426-428`, and the residual is similarly converted in `deepks/deephf/adjoint.py:458-464`. `AdjointResult` construction then applies `_immutable_array` again to the objective gradient, solution, and residual in `deepks/deephf/adjoint.py:507-513`. The final result retains one vector for each semantic value, but construction transiently duplicates already validated immutable solution and residual arrays and can also duplicate the objective gradient.

Required correction: validate and freeze each owned array exactly once, then pass those exact arrays into `AdjointResult`. Add storage-identity or allocation-count tests that distinguish final retained ownership from peak construction copies.

## Approval conditions

Approval requires both P0 findings to be corrected with independent inference, training, provenance, and alias-mutation regressions. The executed force graph must be exactly the graph that was validated, and force-batch acceptance must prove issuance by the configured Reader and the required content-integrity contract.

Approval also requires the physical adjoint residual to remain meaningful under every accepted protocol, selected-response diagnostics to state only reproducible invariants, density construction to be shared within one response-gradient transaction, and compact gradient execution to avoid constructing data that it immediately discards. Restricted and unrestricted call budgets, peak-allocation measurements, and exact selected-versus-full numerical comparisons should cover direct and Z-vector backends.

The adjoint immutable-copy issue is lower severity but remains within the requested duplicate-memory scope and should be resolved or justified with measured negligible impact before final efficiency approval.

## Final assessment

The revision has credible improvements: DFT implementation identity is now protected at the cache boundary, the supported activation domain is differentiable, common leaf replacements are rejected, selected atoms reach the principal lower-level calculations, and ordinary returned gradient objects are materially smaller. It is not ready for approval because execution can still leave the validated model graph, Reader provenance and tensor integrity can still be forged, a declared self-adjoint flag can suppress physical verification, selected responses can carry false diagnostics, and current compact paths retain substantial redundant calculation and peak allocation. The full passing suite demonstrates broad regression stability but does not exercise these reproduced contract and allocation failures.
