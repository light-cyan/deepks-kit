# P4C RKS DeePHF Z-Vector Inference

## 1. Status and objective

P4C implements strict scalar-adjoint inference for the molecular closed-shell pure-LDA RKS domain established by the [P4B direct oracle](./p4b_rks_direct_oracle.md) and the shared conventions in the [P0 scientific contract](./p0_scientific_contract.md).

The backend evaluates the exact analytic nuclear gradient of the accepted finite-grid energy `e_tot = e_base + e_corr` for one scalar correction objective with one literal transpose solve. It retains the explicit, objective-metric, fixed-grid, grid-coordinate, grid-weight, adjoint-metric, occupied-virtual, response, correction, native-reference, and total gradient partitions.

The scalar adjoint is model-specific inference state. The model-independent coordinate-wise `density_response` and `dq_dR_relaxed` remain products of the P4B direct oracle.

## 2. Implemented package boundary

`deepks.deephf.rks_method.RKSDeePHF` owns the independent direct and scalar-adjoint option namespaces, complete-model sensitivity evaluation, correction AO objective construction, trusted adjoint provenance, and explicit backend dispatch.

`deepks.deephf.rks_zvector.RKSDeePHFZVectorGradients` owns fail-closed scalar-adjoint gradient assembly and publishes the native finite-grid, explicit, metric, occupied-virtual, grid-response, correction, and total partitions.

`deepks.deephf.pyscf_rks.RKSAdjointAdapter` owns the characterized PySCF 2.14 and LibXC 7.0.0 RKS response operator, correction-specific transpose solve, induced adjoint density and potential, finite-grid nuclear derivative contractions, provenance, diagnostics, and independent result audit.

`deepks.deephf.adjoint.ScalarAdjointProblem` and `solve_scalar_adjoint` remain reference-neutral. The protocol contains only `dimension`, `dense_operator()`, `apply(vector)`, and `apply_transpose(vector)`; all RKS, LibXC, grid, orbital, and nuclear-coordinate semantics remain in the RKS compatibility adapter.

PySCF response, NumInt, LibXC, grid-generator, grid-weight derivative, native-gradient, and private molecular state access remains isolated in `deepks.deephf.pyscf_rks`.

## 3. Accepted RKS, functional, grid, descriptor, and model state

The scalar-adjoint backend accepts exactly the P4B strict native closed-shell pure-LDA RKS domain. The reference must pass `validate_rks_reference`, including exact native type and convergence, canonical complete real `float64` orbitals and occupations, finite-grid Fock and energy consistency, the characterized normalized `LDA_X + LDA_C_VWN` LibXC 7.0.0 semantics, and the deterministic unpruned `(20, 50)` atom-centered grid contract.

The grid contract binds canonical PySCF 2.14 radial and Becke definitions, canonical radii content, exact cached coordinates and host ownership, cached energy-grid weights, response-generator identity, host-block boundaries, translational grid-weight derivatives, and their independent nuclear-coordinate finite-difference audit.

The occupied-virtual CPKS operator must pass the configured orbital-gap, dimension, finite-value, symmetry, positive-stability, and spectral-condition gates before the transpose solve.

The correction model must satisfy the shared strict force contract: one real finite scalar output, real double-precision finite state, evaluation mode, compatible projector and feature count, deterministic complete energy and sensitivity, unchanged RNG and model state, descriptor-space finite-difference agreement, and accepted ordered-spectrum differentiability.

The method binds the exact reference, molecule, descriptor, projector molecule, projector metadata, model state, functional, and grid state throughout one force transaction.

## 4. Scalar objective and physical RKS operator

Let `s[I,k] = partial e_corr / partial q[I,k]` and define the symmetric AO objective potential

```text
W[mu,nu] = sum_I,k s[I,k] dq_dP[I,k,mu,nu].
```

With accepted closed-shell occupations `n_i = 2` and `W_mo = C.T W C`, the bilateral occupied-virtual objective derivative is

```text
b[a,i] = n_i (W_mo[a,i] + W_mo[i,a]) = 4 W_mo[a,i].
```

For a trial occupied-virtual amplitude `X`, the spin-summed AO density variation and pure-LDA induced potential are

```text
delta P(X) = C_v X (C_o diag(n_i)).T + C_o diag(n_i) X.T C_v.T
G_RKS[delta P] = J[delta P] + K_xc[delta P].
```

`K_xc` is the dense contraction of the accepted LibXC LDA kernel `f_xc` on the exact finite grid. The unshifted physical action is

```text
(A X)[a,i] = (epsilon_a - epsilon_i) X[a,i] + (C_v.T G_RKS[delta P(X)] C_o)[a,i].
```

The adapter materializes this operator up to `operator_dimension_limit`, independently compares its Coulomb plus dense-LDA-`f_xc` action with the characterized native RKS response action, proves symmetry, audits its eigenspectrum and condition number, and exposes it through `ScalarAdjointProblem`.

## 5. One transpose solve and complete gradient decomposition

For one scalar correction energy, the reference-neutral solver performs exactly one dense solve

```text
A.T z = b.
```

Define the adjoint AO density and induced potential

```text
D_z = C_v z (C_o diag(n_i)).T + C_o diag(n_i) z.T C_v.T
V_z = G_RKS[D_z].
```

For nuclear coordinate `R`, let `S_oo^R = C_o.T S^R C_o`, `Wbar_oo = 0.5 (W_oo + W_oo.T)`, and partition the complete finite-grid Hamiltonian derivative as

```text
H^R = H_fixed_grid^R + H_xc_grid_coordinate^R + H_xc_grid_weight^R.
```

The implemented correction-gradient partitions are

```text
g_explicit^R = sum_I,k s[I,k] dq_dR_explicit[R,I,k]
g_metric^R = -2 S_oo^R : Wbar_oo
g_adjoint_fixed_grid^R = -z : (H_fixed_grid,vo^R - S_vo^R epsilon_o)
g_adjoint_grid_coordinate^R = -z : H_xc_grid_coordinate,vo^R
g_adjoint_grid_weight^R = -z : H_xc_grid_weight,vo^R
g_adjoint_nuclear^R = g_adjoint_fixed_grid^R + g_adjoint_grid_coordinate^R + g_adjoint_grid_weight^R
g_adjoint_metric^R = 0.5 S_oo^R : V_z,oo
g_occupied_virtual^R = g_adjoint_nuclear^R + g_adjoint_metric^R
g_response^R = g_metric^R + g_occupied_virtual^R
g_corr^R = g_explicit^R + g_response^R
g_tot^R = g_RKS_native^R + g_corr^R.
```

The native finite-grid reference gradient is independently retained as

```text
g_RKS_native = g_without_grid_response + g_xc_grid_coordinate + g_xc_grid_weight.
```

The objective-metric term is the direct contraction `W : P_metric^R`, while the adjoint-metric term is algebraically identical to the induced-potential contribution `-z : G_RKS[P_metric^R]_vo`. Both are required for equality with the P4B complete density-response oracle.

## 6. Runtime API and result contract

The following P4C symbols are exported from `deepks.deephf`: `RKSAdjoint`, `RKSAdjointAdapter`, `RKSAdjointDiagnostics`, `RKSAdjointError`, and `RKSDeePHFZVectorGradients`.

| API | Result |
|---|---|
| `RKSDeePHF(..., response_options=..., adjoint_options=...)` | One perturbative method with independent direct and scalar-adjoint option namespaces. |
| `RKSDeePHF.adjoint(**options)` | One immutable correction-specific `RKSAdjoint`. |
| `RKSDeePHF.nuc_grad_method(backend="zvector", **options)` | One `RKSDeePHFZVectorGradients` driver. |
| `RKSDeePHF.gradient(backend="zvector", **options)` | Complete `d(e_base + e_corr)/dR`. |
| `RKSDeePHF.forces(backend="zvector", **options)` | Exact negative of the complete gradient. |
| `RKSDeePHFZVectorGradients.kernel(atmlst=None)` | Complete or selected-atom scalar-adjoint gradient. |
| `RKSAdjointAdapter.solve(W)` | One audited `A.T z = b` solve and complete nuclear contraction. |
| `RKSAdjointAdapter.audit_adjoint(adjoint, W)` | Independent equation, provenance, integrity, and partition audit without another solve. |

`backend="direct"` remains the default and constructs the P4B coordinate-wise response. `backend="zvector"` is explicit and returns model-specific scalar-inference state. Any other backend is rejected.

The Z-vector driver publishes `reference_gradient`, `reference_gradient_without_grid_response`, `reference_gradient_xc_grid_coordinate`, `reference_gradient_xc_grid_weight`, `dq_dR_explicit`, `correction_gradient_explicit`, `correction_gradient_metric`, `correction_gradient_adjoint_fixed_grid`, `correction_gradient_adjoint_grid_coordinate`, `correction_gradient_adjoint_grid_weight`, `correction_gradient_adjoint_nuclear`, `correction_gradient_adjoint_metric`, `correction_gradient_occupied_virtual`, `correction_gradient_response`, `correction_gradient`, `de_full`, and the selected `de`.

`RKSAdjoint` contains the exact reference identity and state fingerprint, functional and grid provenance, operator and result-integrity fingerprints, `W`, `b`, `z`, literal-transpose, independent-transpose, and physical residual arrays, `D_z`, `V_z`, every correction-response gradient partition, and `RKSAdjointDiagnostics`. Every numerical result array is immutable real finite `numpy.float64`.

The Z-vector result exposes no coordinate-wise density response, first-order density, response descriptor Jacobian, or relaxed descriptor Jacobian.

## 7. Option namespaces and diagnostics

The direct `response_options` namespace remains the P4B coordinate-wise CPKS configuration. The independent `adjoint_options` namespace contains the following controls.

| Option | Default | Acceptance meaning |
|---|---:|---|
| `residual_tolerance` | `1e-9` | Maximum literal-transpose, independent-transpose, and physical residual. |
| `invariant_tolerance` | `1e-9` | Maximum induced-potential and nuclear-Hamiltonian reconstruction residual. |
| `orbital_gap_tolerance` | `1e-7` | Strict lower bound for the occupied-virtual gap in `Eh`. |
| `operator_stability_tolerance` | `1e-6` | Strict lower bound for the minimum physical CPKS eigenvalue in `Eh`. |
| `operator_condition_tolerance` | `1e8` | Maximum accepted physical CPKS condition number. |
| `operator_symmetry_tolerance` | `1e-10` | Maximum accepted dense-operator symmetry residual. |
| `operator_dimension_limit` | `512` | Maximum occupied-virtual dimension admitted to the dense audit. |
| `objective_symmetry_tolerance` | `1e-10` | Maximum accepted AO-objective, adjoint-density, and adjoint-potential symmetry residual. |

`RKSAdjointDiagnostics` records the PySCF and LibXC versions, normalized functional components, finite-grid fingerprints and point count, orbital gap, every control, operator dimension and spectrum, condition number, symmetry and induced-potential reconstruction residuals, fixed-grid XC and complete Hamiltonian reconstruction residuals, objective and adjoint symmetry residuals, exact solver identity and solve count, objective and solution norms, and maximum plus RMS values for all three equation residuals.

The literal residual is `dense_operator().T @ z - b`. The independent transpose and physical residuals are evaluated through isolated immutable protocol actions after the solve; the accepted symmetric physical RKS operator makes the separately applied physical action the transpose action while the generic protocol retains a literal nonsymmetric transpose test.

## 8. Fail-closed lifecycle

The method stores the exact producing adapter, original adjoint integrity fingerprint, immutable sensitivity fingerprint, descriptor diagnostics, and complete model fingerprint for the current scalar-adjoint evaluation. Consumption requires exact object identity, unchanged original and recomputed integrity, the same reference, functional, grid, model, descriptor, projector, objective, controls, operator, and reconstructed equations.

Every Z-vector driver call clears all public results and trusted adjoint state before work begins. It publishes only after the native complete grid-response gradient, explicit descriptor derivative, scalar adjoint, every partition, and all state checks succeed; any exception clears the result and trusted state again.

The Z-vector path calls neither `RKSResponseAdapter.solve`, `RKSDeePHF.response`, `first_order_density`, `dq_dR_response`, `dq_dR_relaxed`, nor `RKSDeePHFGradients`. It constructs one `(n_ao, n_ao)` adjoint density and induced potential but never a `(n_raw_atom, 3, n_ao, n_ao)` coordinate-wise response.

An unsupported reference, functional, grid, orbital state, response operator, projector, descriptor, or model fails before publication. An invalid control, nonsymmetric objective, failed or nonfinite transpose solve, excessive residual, corrupted grid derivative, inconsistent nuclear partition, foreign or changed adjoint, changed scientific state, or invalid driver binding also fails explicitly. No failure returns a direct, fixed-density, no-grid-response, explicit-only, or stale result.

Zero and coordinate-independent constant corrections retain one audited zero-RHS scalar solve, produce zero correction-gradient partitions, and reduce the total gradient to the complete native grid-response RKS gradient.

## 9. Direct-oracle and force-data boundaries

The P4B direct backend remains the correctness oracle and the sole RKS facility that returns complete coordinate-wise `P^R`, `dq_dR_response`, and `dq_dR_relaxed`. Direct and Z-vector backends remain separately selectable and inspectable on the same method.

The strict RHF force-data producer continues to select its RHF direct backend and persistent relaxed-Jacobian schema. An RKS scalar adjoint contains only one model-specific correction contraction and cannot be consumed as a force-training Jacobian.

The RHF direct and scalar-adjoint paths, RHF scanner, UHF direct and scalar-adjoint paths, RKS direct path, force-training workflow, and energy-only method behavior retain their established package and numerical contracts.

## 10. Deterministic acceptance

The numerical oracle uses a symmetry-disabled distorted neutral `H2O/STO-3G` singlet, normalized `LDA_X + LDA_C_VWN`, the exact prebuilt unpruned `(20, 50)` grid, projector shells `[[0, [0.8, 1.0]], [1, [0.3, 1.0]]]`, and a deterministic nontrivial double-precision nonlinear correction model.

Independent raw-AO Coulomb integrals and dense LibXC quadrature reconstruct the complete `J + f_xc` operator, bilateral objective, literal transpose solution, adjoint density and potential, AO metric formulas, fixed-grid AO/XC, grid-coordinate, and grid-weight contractions without using the production direct response.

Three central-difference steps `(1e-3, 3e-4, 1e-4) Bohr` use independently converged fresh finite-grid RKS references for every Cartesian displacement and validate the complete `e_base + e_corr` gradient.

The acceptance suite also verifies direct-versus-Z agreement for every common partition, nonzero and omission-sensitive Coulomb, `f_xc`, fixed-grid, grid-coordinate, grid-weight, objective-metric, and adjoint-metric terms, a nonsymmetric reference-neutral transpose problem, exactly one scalar solve, forbidden direct and coordinate-wise response paths, zero and constant corrections, atom selection and force sign, control and operator faults, foreign and changed result state, complete failure cleanup, and package ownership.

Final Python 3.11 verification completed with 96 P4C tests, 143 RKS direct tests, 115 UHF direct tests, 152 RHF Z-vector tests, 113 RHF analytic-force tests, 99 force-training tests, 45 baseline tests, and 763 tests in the complete repository suite. Locked dependency synchronization, source and wheel builds, bytecode compilation, architecture checks, and `git diff --check` also completed successfully.

Run the complete verification sequence from the repository root:

```bash
uv sync --locked --python 3.11
uv run pytest tests/rks_zvector_inference
uv run pytest tests/rks_analytic_forces
uv run pytest tests/uhf_analytic_forces
uv run pytest tests/zvector_inference
uv run pytest tests/analytic_forces
uv run pytest tests/force_training
uv run pytest tests/baseline
uv run pytest
uv build
git diff --check
```
