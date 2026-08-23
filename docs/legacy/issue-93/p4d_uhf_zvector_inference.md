# P4D UHF DeePHF Z-Vector Inference

## 1. Status and objective

P4D implements strict scalar-adjoint inference for the molecular unrestricted Hartree-Fock domain established by the [P4A direct oracle](./p4a_uhf_direct_oracle.md) and the shared conventions in the [P0 scientific contract](./p0_scientific_contract.md).

The backend evaluates the exact analytic nuclear gradient of `e_tot = e_base + e_corr` for one scalar correction objective with one coupled alpha/beta transpose solve. It retains alpha, beta, occupied-virtual, AO-metric, explicit, response, correction, native-reference, and total gradient partitions.

The scalar adjoint is model-specific inference state. The model-independent coordinate-wise alpha, beta, and total density responses and `dq_dR_relaxed` remain products of the P4A direct oracle.

## 2. Implemented package boundary

`deepks.deephf.uhf_method.UHFDeePHF` owns independent direct and scalar-adjoint option namespaces, complete-model sensitivity evaluation, correction AO objective construction, trusted adjoint provenance, and explicit backend dispatch.

`deepks.deephf.uhf_zvector.UHFDeePHFZVectorGradients` owns fail-closed scalar-adjoint gradient assembly and publishes the spin-resolved and spin-summed explicit, metric, occupied-virtual, response, correction, native-reference, and total partitions.

`deepks.deephf.pyscf_uhf.UHFAdjointAdapter` owns the characterized PySCF 2.14 coupled UHF response operator, correction-specific transpose solve, induced alpha/beta adjoint densities and potentials, nuclear derivative contractions, provenance, diagnostics, and independent result audit.

`deepks.deephf.adjoint.ScalarAdjointProblem` and `solve_scalar_adjoint` remain reference-neutral. The protocol contains only `dimension`, `dense_operator()`, `apply(vector)`, and `apply_transpose(vector)`; all UHF spin, orbital, integral, and nuclear-coordinate semantics remain in the UHF compatibility adapter.

PySCF UHF Hessian, UC-PHF, J/K, and private molecular state access remains isolated in `deepks.deephf.pyscf_uhf`.

## 3. Accepted UHF, descriptor, and model state

The scalar-adjoint backend accepts exactly the strict native molecular UHF domain defined by P4A. The reference must pass `validate_uhf_reference`, including exact native type and convergence, complete canonical real `numpy.float64` alpha and beta orbitals, zero-or-one Aufbau occupations consistent with `mol.nelec` and `mol.spin`, valid AO densities and native J/K potentials, reconstructed energy consistency, and unchanged molecular and orbital provenance.

Both spin channels must contain occupied and virtual orbitals with accepted gaps. The full coupled alpha/beta occupied-virtual operator must pass the configured dimension, finite-value, symmetry, positive-stability, and spectral-condition gates before the transpose solve.

The correction model must satisfy the shared strict force contract: one real finite scalar output, real double-precision finite state, evaluation mode, compatible projector and feature count, deterministic complete energy and sensitivity, unchanged RNG and model state, descriptor-space finite-difference agreement, and accepted ordered-spectrum differentiability.

The method binds the exact reference, molecule, descriptor, projector molecule, projector metadata, and model state throughout one force transaction.

## 4. Coupled scalar objective and UHF operator

Let `s[I,k] = partial e_corr / partial q[I,k]` and define the symmetric spin-summed AO objective potential

```text
W[mu,nu] = sum_I,k s[I,k] dq_dP[I,k,mu,nu].
```

Because the descriptor consumes `P_alpha + P_beta`, the same `W` acts on both spin densities. With `W_sigma = C_sigma.T W C_sigma` and accepted UHF occupations `n_sigma,i = 1`, the bilateral occupied-virtual objective derivatives are

```text
b_sigma[a,i] = n_sigma,i (W_sigma[a,i] + W_sigma[i,a]) = 2 W_sigma[a,i].
```

For trial occupied-virtual amplitudes `X_alpha` and `X_beta`, define

```text
delta P_sigma(X_sigma) = C_sigma,v X_sigma C_sigma,o.T + C_sigma,o X_sigma.T C_sigma,v.T
G_alpha = J[delta P_alpha + delta P_beta] - K[delta P_alpha]
G_beta = J[delta P_alpha + delta P_beta] - K[delta P_beta].
```

The unshifted physical coupled action is

```text
(A X)_sigma[a,i] = (epsilon_sigma,a - epsilon_sigma,i) X_sigma[a,i] + (C_sigma,v.T G_sigma C_sigma,o)[a,i].
```

The adapter materializes the complete alpha/beta operator up to `operator_dimension_limit`, proves symmetry, audits the eigenspectrum and condition number, fingerprints the operator, and exposes the concatenated amplitudes through one `ScalarAdjointProblem`.

## 5. One transpose solve and complete gradient decomposition

For one scalar correction energy, the reference-neutral solver performs exactly one coupled solve

```text
A.T [z_alpha, z_beta] = [b_alpha, b_beta].
```

Define one adjoint AO density and induced potential in each spin channel:

```text
D_z,sigma = C_sigma,v z_sigma C_sigma,o.T + C_sigma,o z_sigma.T C_sigma,v.T
V_z,alpha = J[D_z,alpha + D_z,beta] - K[D_z,alpha]
V_z,beta = J[D_z,alpha + D_z,beta] - K[D_z,beta].
```

For nuclear coordinate `R`, let `S_sigma,oo^R = C_sigma,o.T S^R C_sigma,o` and `Wbar_sigma,oo = 0.5 (W_sigma,oo + W_sigma,oo.T)`. The implemented spin-channel partitions are

```text
g_metric,sigma^R = -S_sigma,oo^R : Wbar_sigma,oo
g_adjoint_nuclear,sigma^R = -z_sigma : (H_sigma,vo^R - S_sigma,vo^R epsilon_sigma,o)
g_adjoint_metric,sigma^R = 0.5 S_sigma,oo^R : V_z,sigma,oo
g_occupied_virtual,sigma^R = g_adjoint_nuclear,sigma^R + g_adjoint_metric,sigma^R.
```

The spin-summed identities are

```text
g_metric = sum_sigma g_metric,sigma
g_occupied_virtual = sum_sigma g_occupied_virtual,sigma
g_response = g_metric + g_occupied_virtual
g_corr = g_explicit + g_response
g_tot = g_UHF_native + g_corr.
```

The objective-metric spin terms equal the direct contractions `W : P_metric,sigma^R`. The alpha and beta occupied-virtual adjoint values are coupled equation-channel contractions; their sum, rather than each channel separately, is the quantity that equals the P4A direct occupied-virtual contraction.

## 6. Runtime API and result contract

The following P4D symbols are exported from `deepks.deephf`: `UHFAdjoint`, `UHFAdjointAdapter`, `UHFAdjointDiagnostics`, `UHFAdjointError`, and `UHFDeePHFZVectorGradients`.

| API | Result |
|---|---|
| `UHFDeePHF(..., response_options=..., adjoint_options=...)` | One perturbative UHF method with independent direct and scalar-adjoint option namespaces. |
| `UHFDeePHF.adjoint(**options)` | One immutable correction-specific coupled `UHFAdjoint`. |
| `UHFDeePHF.nuc_grad_method(backend="zvector", **options)` | One `UHFDeePHFZVectorGradients` driver. |
| `UHFDeePHF.gradient(backend="zvector", **options)` | Complete `d(e_base + e_corr)/dR`. |
| `UHFDeePHF.forces(backend="zvector", **options)` | Exact negative of the complete gradient. |
| `UHFDeePHFZVectorGradients.kernel(atmlst=None)` | Complete or selected-atom scalar-adjoint gradient. |
| `UHFAdjointAdapter.solve(W)` | One audited coupled `A.T z = b` solve and complete nuclear contraction. |
| `UHFAdjointAdapter.audit_adjoint(adjoint, W)` | Independent equation, provenance, integrity, and partition audit without another solve. |

`backend="direct"` remains the default and constructs the P4A coordinate-wise response. `backend="zvector"` is explicit and returns model-specific scalar-inference state.

The Z-vector driver publishes `reference_gradient`, `dq_dR_explicit_spin`, `dq_dR_explicit`, every spin and total correction partition, `de_full`, and the selected `de`.

`UHFAdjoint` contains exact reference and operator provenance, `W`, alpha and beta `b`, `z`, solver, independent-transpose, and physical residual arrays, alpha and beta adjoint AO densities and potentials, every correction-response gradient partition, and `UHFAdjointDiagnostics`. Every numerical result array is immutable real finite `numpy.float64`.

## 7. Option namespaces and diagnostics

The direct `response_options` namespace remains the P4A coordinate-wise UC-PHF configuration. The independent `adjoint_options` namespace contains the following controls.

| Option | Default | Acceptance meaning |
|---|---:|---|
| `residual_tolerance` | `1e-9` | Maximum solver, independent-transpose, and physical residual. |
| `invariant_tolerance` | `1e-9` | Maximum induced-potential and nuclear-gradient reconstruction residual. |
| `orbital_gap_tolerance` | `1e-7` | Strict lower bound for both spin occupied-virtual gaps in `Eh`. |
| `operator_stability_tolerance` | `1e-6` | Strict lower bound for the minimum coupled UHF eigenvalue in `Eh`. |
| `operator_condition_tolerance` | `1e8` | Maximum accepted coupled-operator condition number. |
| `operator_symmetry_tolerance` | `1e-10` | Maximum accepted dense-operator symmetry residual. |
| `operator_dimension_limit` | `512` | Maximum combined alpha/beta occupied-virtual dimension admitted to the dense audit. |
| `objective_symmetry_tolerance` | `1e-10` | Maximum accepted AO-objective, adjoint-density, and adjoint-potential symmetry residual. |

`UHFAdjointDiagnostics` records the PySCF version, both spin gaps and dimensions, every control, complete operator dimension and spectrum, condition number, operator and objective symmetry residuals, alpha and beta adjoint-density and potential symmetry residuals, gradient reconstruction residual, solver identity and solve count, objective and solution norms, and maximum plus RMS values for all three equation residuals.

## 8. Fail-closed lifecycle and direct boundary

The method stores the exact producing adapter, original adjoint integrity fingerprint, immutable sensitivity fingerprint, descriptor diagnostics, complete model fingerprint, and adapter controls for the current scalar-adjoint evaluation. Consumption requires exact object identity and unchanged provenance, integrity, controls, reference, model, descriptor, projector, objective, operator, equations, and partitions.

Every Z-vector driver call clears all public results and trusted adjoint state before work begins. It publishes only after the native UHF gradient, explicit descriptor derivative, coupled scalar adjoint, every partition, and all state checks succeed; any exception clears the result and trusted state again.

The Z-vector path performs one neutral scalar-adjoint solve and constructs one alpha and one beta adjoint AO density. The P4A direct backend remains the coordinate-wise source of alpha, beta, and total `P^R`, descriptor response, and relaxed descriptor Jacobian used as the scientific oracle.

The strict RHF force-data producer continues to select its RHF direct backend and persistent relaxed-Jacobian schema. A UHF scalar adjoint contains one model-specific correction contraction and carries no force-training Jacobian semantics.

Zero and coordinate-independent constant corrections retain one audited zero-RHS scalar solve, produce zero correction-gradient partitions, and reduce the total gradient to the native UHF gradient.

## 9. Deterministic acceptance

The primary numerical oracle uses a symmetry-disabled distorted neutral `NH2/STO-3G` doublet, projector shells `[[0, [0.8, 1.0]], [1, [0.3, 1.0]]]`, and a deterministic nontrivial double-precision nonlinear correction model.

Independent raw-AO Coulomb and exchange integrals reconstruct the complete coupled operator, bilateral alpha/beta objectives, literal transpose solution, adjoint densities and potentials, AO metric formulas, and nuclear contractions without using the production direct response.

Three central-difference steps `(1e-3, 3e-4, 1e-4) Bohr` use independently converged fresh UHF references for every Cartesian displacement and validate the complete `e_base + e_corr` gradient. A distinct bent `BH2/STO-3G` doublet validates all nine Cartesian coordinates against both a fresh direct method and total-energy finite differences.

The acceptance suite also verifies direct-versus-Z agreement, nonzero and omission-sensitive cross-spin, alpha, beta, metric, and occupied-virtual terms, a nonsymmetric reference-neutral transpose problem, exactly one scalar solve, direct-response and UC-PHF exclusion, zero and constant corrections, atom selection and force sign, independent option namespaces, control and operator faults, foreign and changed result state, complete failure cleanup, and package ownership.

Final Python 3.11 verification completed with 40 P4D tests, 115 UHF direct tests, 96 RKS Z-vector tests, 143 RKS direct tests, 152 RHF Z-vector tests, 114 RHF analytic-force tests, 99 force-training tests, 45 baseline tests, and 804 tests in the complete repository suite. Locked dependency synchronization, source and wheel builds, bytecode compilation, architecture checks, and `git diff --check` also completed successfully.

Run the complete verification sequence from the repository root:

```bash
uv sync --locked --python 3.11
uv run pytest tests/uhf_zvector_inference
uv run pytest tests/uhf_analytic_forces
uv run pytest tests/rks_zvector_inference
uv run pytest tests/rks_analytic_forces
uv run pytest tests/zvector_inference
uv run pytest tests/analytic_forces
uv run pytest tests/force_training
uv run pytest tests/baseline
uv run pytest
uv build
git diff --check
```
