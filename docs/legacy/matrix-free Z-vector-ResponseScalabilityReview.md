# Response Scalability and Correctness Review

## Review scope

This review assesses the current working-tree implementation of occupied-virtual response diagnostics, scalar adjoint solves, coordinate-blocked RHF gradients, force-data provenance, and the associated scalability tests as of 2026-08-23.

## Conclusion

The dense-response scalability problem is partially addressed for RHF, but it is not resolved across the project. RHF now has a matrix-free GMRES adjoint solve, a large-dimension matrix-free diagnostic path, and coordinate-blocked direct response processing. UHF and RKS still materialize dense response matrices, enforce the default dimension limit of 512, and use dense adjoint solves; UKS uses the UHF response and adjoint infrastructure and therefore retains the same limitation.

The current large-dimension RHF diagnostic also has a correctness defect: a fixed 16-step Lanczos projection is treated as conclusive evidence of positive definiteness and acceptable conditioning. Its smallest Ritz value is not a certified lower bound for the smallest eigenvalue of the full operator, so an unstable operator can be accepted as stable.

## Current implementation status

| Area | Current behavior | Assessment |
| --- | --- | --- |
| Shared scalar adjoint | Supports matrix-free `LinearOperator` and GMRES, with optional dense solving. | Useful foundation. |
| RHF adjoint | Uses an action-only scalar problem and GMRES. | Dense solve removed from this path. |
| RHF operator diagnostics above the dimension limit | Uses sampled symmetry checks and at most 16 Lanczos steps. | Avoids dense materialization, but the stability and condition gates are not numerically certified. |
| RHF operator diagnostics at or below the dimension limit | Builds the complete response matrix and runs `eigvalsh`, although the subsequent adjoint solve uses GMRES. | Retains redundant cubic work in the default path. |
| RHF coordinate response | Supports atom-coordinate blocks and does not retain the complete AO density response. | Reduces peak transient memory, while the final descriptor response and gradient arrays remain full-size result data. |
| UHF response and adjoint | Builds the complete coupled response matrix, rejects dimensions above 512 by default, and selects the dense adjoint solver. | Scalability problem remains. |
| RKS response and adjoint | Builds the complete response matrix, rejects dimensions above 512 by default, and selects the dense adjoint solver. | Scalability problem remains. |
| UKS response and adjoint | Reuses the UHF response and adjoint infrastructure with UKS-specific response actions. | UHF scaling limitations propagate to UKS. |

## Confirmed findings

### P1: The RHF matrix-free stability audit can falsely accept an unstable operator

`_matrix_free_response_operator_diagnostics` constructs a Krylov subspace with at most 16 Lanczos steps and applies the minimum and maximum eigenvalues of the projected tridiagonal matrix as hard stability and condition-number gates. The smallest Ritz value obtained from this subspace can exceed the true smallest eigenvalue, and the implementation does not calculate a Ritz residual, convergence bound, or other error enclosure.

A direct reproduction with the current algorithm used a diagonal operator of dimension 1000 with an exact minimum eigenvalue of `-1.0e-8` and remaining eigenvalues distributed from 1 to 10. The 16-step estimate reported a minimum eigenvalue of approximately `1.0586515400` and a condition estimate of approximately `9.4180497174`, so the unstable operator passed the current audit.

This is a correctness defect rather than only a diagnostic-quality limitation because the estimates directly authorize production calculation. The safety gate needs a converged extremal-eigenvalue method with residual-based error bounds and conservative failure on unresolved uncertainty, or the estimate must become non-authoritative telemetry.

### P1: UHF, RKS, and UKS retain dense cubic response paths

The UHF and RKS adapters allocate an identity matrix and a complete response matrix, apply the operator to basis-vector batches, and use dense symmetric eigensolves. Both reject response dimensions above `operator_dimension_limit`, which defaults to 512. Their adjoint adapters explicitly request `solver="dense"`. UKS derives its internal response and adjoint adapters from the UHF infrastructure, so the same dense construction and solve remain active there.

Consequently, the project-wide memory growth, cubic eigensolve cost, and fixed-dimension rejection are still present outside RHF.

### P2: The default RHF path still performs a redundant dense audit for dimensions up to 512

RHF switches to matrix-free diagnostics only when the occupied-virtual dimension is greater than `operator_dimension_limit`. At or below the default limit of 512, it constructs the complete matrix and performs `eigvalsh`; the matrix is then discarded because the adjoint is solved separately with GMRES. This preserves the original dense cost for common small and medium calculations and duplicates operator work.

The production path should use action-only diagnostics and solving for every dimension. Exact dense reconstruction can remain an explicit debug or validation facility rather than a size-dependent production branch.

### P2: Selecting atoms does not reduce blocked-response computation

The gradient driver validates `atmlst`, but `_blocked_response` calls `coordinate_blocks(block_size)` without passing the selected atoms and processes every atom. The driver slices `de_full` only after all response blocks, descriptor derivatives, and gradient partitions have been computed.

This behavior returns the correct selected output but fails to realize the expected execution reduction for partial gradients. The selected atom indices need to constrain coordinate-block generation and downstream allocations while preserving full-result behavior when no selection is supplied.

### P2: Execution block settings alter scientific dataset compatibility

The force-data compatibility seed includes the complete `response` mapping. The mapping now contains `coordinate_block_size` and `response_block_count`, so scientifically equivalent force data generated with different chunking are assigned different compatibility fingerprints. `response_block_count` can also vary with atom count, making compatible frames unsuitable for grouping solely because of an execution detail.

Block settings belong in execution provenance but should be excluded from the scientific compatibility seed. Schema validation should also verify that block metadata is internally consistent when it is recorded.

### P2: A supplied matrix-free fingerprint bypasses operator-action verification

`_matrix_free_operator_fingerprint` returns a supplied fingerprint immediately and skips its deterministic action probes. The RHF scalar problem supplies a digest of coefficients, energies, occupations, and orbital masks, but the digest does not include the actual Coulomb/exchange response action. A response-action implementation change can therefore leave the fingerprint unchanged.

The final fingerprint should combine the supplied state digest with deterministic operator-action probe results instead of choosing one source or the other.

## Test evidence

The focused response, strict-contract, Z-vector, force-generation, and schema tests completed with `205 passed in 35.90s`.

The complete test suite completed with `843 passed in 447.25s`.

These passing results establish regression consistency for the exercised cases but do not cover the failure modes above. The scalability fixture has an RHF response dimension of four and forces the large-dimension branch by setting `operator_dimension_limit=1`; it does not exercise a naturally large response, an allocation bound, or the default threshold. The suite also lacks an adversarial-spectrum test for the stability audit, a selected-atom work-reduction test, a block-size compatibility test, and large-dimension UHF, RKS, and UKS coverage.

## Recommended correction order and acceptance criteria

1. Replace the RHF fixed-step stability gate with a conservative, convergence-aware method; add an adversarial operator test that contains a weakly represented negative mode and must be rejected.
2. Convert UHF, RKS, and UKS response diagnostics and adjoint solves to action-only iterative paths; verify that production execution does not allocate a square response matrix and does not reject a valid calculation solely because its response dimension exceeds 512.
3. Remove the automatic dense RHF diagnostic branch from production execution; retain exact dense reconstruction only behind an explicit validation option.
4. Propagate selected atom indices into RHF coordinate-block generation and allocate only the selected result extent; verify operator and derivative call counts as well as numerical output.
5. Separate scientific compatibility fields from execution provenance and validate recorded block metadata; verify that changing block size does not change compatibility for identical scientific inputs.
6. Combine state and action evidence in matrix-free operator fingerprints; verify that a changed operator action changes the fingerprint even when orbital state arrays are unchanged.
7. Add genuine large-response tests or allocation sentinels for every supported reference family while keeping the default local suite compact.
