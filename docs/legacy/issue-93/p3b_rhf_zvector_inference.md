# P3B RHF DeePHF Z-Vector Inference

## 1. Status and scope

P3B implements strict scalar-adjoint inference for the molecular RHF DeePHF support domain defined by the [P0 scientific contract](./p0_scientific_contract.md) and validated by the [P2 direct oracle](./p2_rhf_direct_oracle.md).

The Z-vector backend evaluates the exact analytic nuclear gradient of `e_tot = e_base + e_corr` for one scalar correction objective with one transpose response solve. It preserves the explicit, AO-metric, occupied-virtual, and complete response partitions and is numerically equivalent to the P2 direct backend inside the accepted domain.

P3B also supplies a reference-neutral scalar-adjoint protocol and a strict fresh-reference RHF gradient scanner. The protocol is an extension boundary for reference-specific adapters; the implemented scientific backend in this phase is molecular RHF.

The P3A persistent force-training derivative remains the model-independent complete `dq_dR_relaxed` produced by the direct backend. The scalar Z-vector result is model-specific and is an inference gradient, not a replacement source for a stored relaxed descriptor Jacobian.

## 2. Indices, axes, signs, and units

Let `A` denote a raw atom, `x` a Cartesian component in `(x, y, z)` order, `I` a descriptor atom, `k` a descriptor feature, `mu` and `nu` AO indices, `i` and `j` occupied RHF orbitals, and `a` and `b` virtual RHF orbitals. Nuclear coordinates are in `Bohr`, energies are in `Eh`, energy gradients are in `Eh/Bohr`, and forces are the negative gradients.

| Quantity | Runtime axes | Unit | Meaning |
|---|---|---|---|
| `s = partial e_corr / partial q` | `(descriptor_atom, feature)` | `Eh` | Complete model sensitivity after every active model branch. |
| `dq_dP` | `(descriptor_atom, feature, ao, ao)` | `1` | Descriptor derivative with respect to the spin-summed AO density. |
| `W = partial e_corr / partial P` | `(ao, ao)` | `Eh` | Symmetric correction AO objective potential. |
| `X^R` | `(raw_atom, cartesian, virtual, occupied)` | `Bohr^-1` | Direct occupied-virtual orbital response. |
| `A` | `(virtual * occupied, virtual * occupied)` | `Eh` | Unshifted physical RHF occupied-virtual response operator. |
| `B^R` | `(raw_atom, cartesian, virtual, occupied)` | `Eh/Bohr` | Nuclear right-hand side in `A X^R = -B^R`. |
| `b` | `(virtual, occupied)` | `Eh` | Correction-specific adjoint right-hand side. |
| `z` | `(virtual, occupied)` | `1` | Scalar RHF Z-vector satisfying `A.T z = b`. |
| `S^R` | `(raw_atom, cartesian, ao, ao)` | `Bohr^-1` | AO overlap derivative. |
| `h^R` | `(raw_atom, cartesian, ao, ao)` | `Eh/Bohr` | Effective RHF Hamiltonian derivative supplied by the PySCF 2.14 adapter. |
| Correction-gradient partitions | `(raw_atom, cartesian)` | `Eh/Bohr` | Explicit, metric, occupied-virtual, response, and complete correction gradients. |
| `de_full` and `de` | `(raw_atom, cartesian)` or selected rows | `Eh/Bohr` | Complete `d(e_base + e_corr)/dR`, before or after `atmlst` selection. |

All strict adjoint arrays are real `numpy.float64` and finite. Returned forces are exactly the negative of the corresponding gradient.

The response vector uses the C-order flattening of `(virtual, occupied)`, so flat index `a * n_occupied + i` defines both rows and columns of `A`; nuclear derivative arrays lead with `(raw_atom, cartesian)`.

## 3. Correction objective and closed-shell factors

The model sensitivity and AO objective potential are

```text
s[I,k] = partial e_corr / partial q[I,k]
W[mu,nu] = sum_I,k s[I,k] dq_dP[I,k,mu,nu].
```

The adapter requires `W` to be symmetric within `objective_symmetry_tolerance`. With `W_mo = C.T W C` and accepted RHF occupations `n_i = 2`, the bilateral occupied-virtual objective derivative is

```text
b[a,i] = n_i (W_mo[a,i] + W_mo[i,a]) = 4 W_mo[a,i].
```

The bilateral form is required because an occupied-virtual coefficient variation changes the spin-summed AO density on both AO indices. The implementation derives `s` from the complete double-precision model, contracts it with `dq_dP`, and independently tests `W` against PyTorch differentiation of `e_corr` with respect to the AO density.

## 4. Direct response equation and physical operator

For a trial occupied-virtual amplitude `X`, define the closed-shell AO density variation and induced RHF potential by

```text
delta P(X) = C_v X (C_o diag(n_i)).T + C_o diag(n_i) X.T C_v.T
G[delta P] = J[delta P] - 0.5 K[delta P].
```

The unshifted physical response action is

```text
(A X)[a,i] = (epsilon_a - epsilon_i) X[a,i] + (C_v.T G[delta P(X)] C_o)[a,i].
```

For the accepted real closed-shell convention, an independently constructed AO-to-MO integral representation is

```text
A[a,i,b,j] = (epsilon_a - epsilon_i) delta_ab delta_ij + 4 (a i|b j) - (a b|i j) - (a j|i b).
```

The adapter explicitly materializes `A` from the physical response action up to `operator_dimension_limit`, proves symmetry within `operator_symmetry_tolerance`, requires its minimum eigenvalue to exceed `operator_stability_tolerance`, and requires its spectral condition number not to exceed `operator_condition_tolerance`. The adjoint solve separately evaluates the literal matrix, transpose-action, and physical-action residuals, while the acceptance test constructs the same operator independently from AO-to-MO two-electron integrals.

The direct occupied-virtual equation can be written as

```text
A X^R = -B^R
B^R = B_bare^R + B_metric^R
B_bare^R[a,i] = h^R[a,i] - epsilon_i S^R[a,i]
B_metric^R[a,i] = (C_v.T G[P_metric^R] C_o)[a,i].
```

The occupied-occupied metric response obeys `U^R_ij + U^R_ji = -S^R_ij` and produces `P_metric^R`. This term is part of the complete basis-aware response even though it is not an occupied-virtual solve amplitude.

## 5. Transpose solve and gradient decomposition

For one scalar correction energy, the reference-neutral solver performs exactly one literal dense solve

```text
A.T z = b.
```

Since `X^R = -A^-1 B^R`, the occupied-virtual correction contraction is `b : X^R = -z : B^R`. The RHF adapter retains the nuclear and metric parts separately instead of hiding the AO-metric contribution inside one response number.

Define the symmetric occupied objective block `Wbar_oo = 0.5 (W_oo + W_oo.T)`, the adjoint AO density, and its induced potential by

```text
D_z = C_v z (C_o diag(n_i)).T + C_o diag(n_i) z.T C_v.T
V_z = G[D_z].
```

For every raw-atom Cartesian coordinate, the implemented closed-shell partitions are

```text
g_explicit^R = sum_I,k s[I,k] dq_dR_explicit[R,I,k]
g_metric^R = -2 sum_i,j S^R[i,j] Wbar_oo[i,j]
g_adjoint_nuclear^R = -sum_a,i z[a,i] B_bare^R[a,i]
g_adjoint_metric^R = 0.5 sum_i,j S^R[i,j] V_z,oo[i,j]
g_occupied_virtual^R = g_adjoint_nuclear^R + g_adjoint_metric^R
g_response^R = g_metric^R + g_occupied_virtual^R
g_corr^R = g_explicit^R + g_response^R
g_tot^R = g_reference^R + g_corr^R.
```

The factor `-2` in `g_metric` is the accepted `n_i = 2` closed-shell specialization of the bilateral objective contraction. The `0.5` adjoint-metric term is the exact contraction corresponding to the metric-induced part of `B^R`; omitting either metric term breaks equivalence to the complete direct AO-density response.

`RHFDeePHFZVectorGradients` publishes these values as `correction_gradient_explicit`, `correction_gradient_metric`, `correction_gradient_adjoint_nuclear`, `correction_gradient_adjoint_metric`, `correction_gradient_occupied_virtual`, `correction_gradient_response`, `correction_gradient`, and `de_full`. It verifies both partition identities before publishing a result.

## 6. Reference-neutral adjoint interface

`deepks.deephf.ScalarAdjointProblem` is a runtime-checkable protocol with `dimension`, `dense_operator()`, `apply(vector)`, and `apply_transpose(vector)`. It expresses only the scalar linear algebra and contains no RHF, orbital, nuclear-coordinate, projector, or model assumptions.

`deepks.deephf.solve_scalar_adjoint(problem, objective_gradient, residual_tolerance=..., require_physical_residual=...)` requires a positive response dimension and exact real finite `float64` vectors and matrix. It solves `A.T z = b` once with `numpy.linalg.solve`, stores immutable arrays, and reports a SHA-256 operator fingerprint and a complete result-integrity fingerprint.

Three residuals remain distinct: `solver_residual = dense_operator().T @ z - b`, `transpose_residual = apply_transpose(z) - b`, and `physical_residual = apply(z) - b`. Each problem action receives an isolated immutable solution snapshot, action outputs are copied immediately, and the response dimension and dense operator must remain unchanged after both independent actions. The literal and independent transpose residuals are always enforced; the RHF adapter also enforces the physical residual because its separately audited operator is symmetric.

`AdjointDiagnostics` records the solver identity, response dimension, solve count, residual tolerance, objective and solution norms, and maximum and RMS values for all three residuals. Reference-specific stability, condition, symmetry, and nuclear-contraction checks belong to the adapter that implements the protocol.

## 7. RHF backend API and lifecycle

`DeePHF.nuc_grad_method()`, `DeePHF.gradient()`, and `DeePHF.forces()` retain `backend="direct"` as their default. `backend="zvector"` is an explicit alternative, and any other backend value raises an error.

The direct driver is `RHFDeePHFGradients` with `backend == "direct"`; the scalar-adjoint driver is `RHFDeePHFZVectorGradients` with `backend == "zvector"`. Both provide `kernel(atmlst=None)`, `run(atmlst=None)`, `forces(atmlst=None)`, and `as_scanner(...)`, while their response result types and backend-specific fields remain distinguishable.

`DeePHF.adjoint(...)` returns one correction-specific `RHFAdjoint`. The Z-vector driver obtains descriptor diagnostics, one internally consistent model sensitivity, and that adjoint before assembling the native RHF and correction-gradient partitions.

Strict force inference evaluates the complete correction scalar and its descriptor sensitivity from the same differentiable model execution, including a validated real finite `float64` element constant when the model provides one. The differentiable scalar must equal the ordinary scalar on the same descriptor with and without Torch gradient recording, repeated evaluations must produce bitwise-identical complete energies and sensitivities, and every sensitivity component must agree with a deterministic descriptor-coordinate central difference using `h = 1e-5 max(1, abs(q))`, `atol = 2e-7`, and `rtol = 2e-5`. Every evaluation also requires unchanged model and scientific-state fingerprints and unchanged Python, NumPy, Torch CPU, and initialized Torch CUDA global RNG states; an evaluation-mode model with execution hooks, stochastic behavior, RNG consumption, detached numerical dependence, gradient-mode-dependent semantics, or persistent parameter, buffer, generator, callable, or semantic-state mutation is rejected.

`DeePHF(..., response_options=..., adjoint_options=...)` stores independent method-level namespaces: `response_options` configures only the direct backend and `adjoint_options` configures only the Z-vector backend. Per-driver keyword options passed to `nuc_grad_method`, `gradient`, `forces`, or `adjoint` override the matching method-level namespace without consulting the other namespace.

The Z-vector option namespace contains `residual_tolerance`, `orbital_gap_tolerance`, `operator_stability_tolerance`, `operator_condition_tolerance`, `operator_symmetry_tolerance`, `operator_dimension_limit`, and `objective_symmetry_tolerance`. The defaults are respectively `1e-9`, `1e-7`, `1e-6`, `1e8`, `1e-10`, `512`, and `1e-10`.

The direct and Z-vector option namespaces are validated independently when their own backend is selected. A direct-only option such as `cphf_tolerance` in `adjoint_options`, a Z-vector-only option such as `objective_symmetry_tolerance` in `response_options`, an unknown per-driver option, or a fallback request is rejected by the affected backend without contaminating the other backend.

Every Z-vector `kernel` call clears all previously published driver results before validation. The driver keeps immutable public bindings to its original method, molecule, and `zvector` backend, while the method binds its exact reference, molecule, descriptor, projector molecule, normalized projector metadata, scientific-state fingerprint, model fingerprint, and internally issued adjoint. A model, descriptor, reference, operator, solve, residual, or assembly failure leaves every result field unset and propagates the original strict error; it never invokes the direct driver and never returns an explicit-only gradient.

The Z-vector path constructs neither the coordinate-wise occupied-virtual response `X^R` nor the complete coordinate-wise AO density response `P^R`. It does not call `response()`, `first_order_density()`, `dq_dR_response()`, `dq_dR_relaxed()`, or the direct gradient driver.

## 8. RHF diagnostics and failure gates

`RHFAdjointDiagnostics` records the minimum occupied-virtual gap, exact PySCF runtime version, every accepted control, response dimension, minimum and maximum operator eigenvalues, condition number, operator symmetry residual, objective symmetry residual, adjoint-density and adjoint-potential symmetry residuals, solver and solve count, objective and solution norms, and maximum and RMS literal, independent-transpose, and physical residuals.

`RHFAdjoint` binds the exact reference object identity and state fingerprint, the operator fingerprint, an integrity fingerprint, `W`, `b`, `z`, all three residual arrays, `D_z`, `V_z`, every response-gradient partition, and the diagnostics. Its numerical arrays are immutable, and the consuming method independently rebuilds the objective, operator actions, residuals, adjoint density and potential, AO derivative contractions, and every gradient partition without performing another solve.

The backend fails explicitly for an unsupported or unconverged RHF reference, missing occupied or virtual space, insufficient occupied-virtual gap, a response dimension above the audit limit, a nonsymmetric, unstable, singular, or ill-conditioned operator, an incompatible, nonfinite, training-mode, hooked, stochastic, or state-mutating model, an incompatible or mutated projector, a nondifferentiable descriptor, an invalid or nonsymmetric objective potential, a failed or nonfinite solve, an excessive independently reproduced residual, a nonfinite or nonsymmetric adjoint intermediate, an invalid gradient shape or dtype, or an inconsistent gradient partition.

| Boundary | Explicit error behavior |
|---|---|
| Generic scalar adjoint | `TypeError`, `ValueError`, or `AdjointError` reports a protocol, dimension, dtype, shape, finite-value, control, solve, or residual violation. |
| RHF capability and adapter | `DeePHFCapabilityError` or `RHFAdjointError` reports a reference, orbital-gap, operator, objective, solve, residual, symmetry, or contraction violation. |
| Z-vector gradient driver | The originating capability, descriptor, model, or adjoint error propagates after every public result field has been cleared. |
| Fresh-reference construction | `TypeError`, `ValueError`, `RHFScannerReferenceError`, or `DeePHFCapabilityError` reports input, static metadata, copied SCF-control, convergence, occupation, or root-continuity failure. |
| Scanner evaluation and publication | `RHFDeePHFScannerError` reports configuration, model-fingerprint, energy, gradient, or state-publication violations while retaining no current result. |

The zero-correction and constant-correction cases still execute the strict adjoint path with `b = 0`; every correction-gradient partition is zero and `de_full` equals the native RHF gradient within the accepted numerical tolerance.

## 9. PySCF 2.14 compatibility boundary

`deepks.deephf.pyscf_rhf` is the isolated compatibility layer for the characterized PySCF `2.14` series. It owns native RHF validation, reference fingerprints and provenance snapshots, AO overlap derivatives, effective Hamiltonian derivatives, Coulomb-exchange response actions, dense occupied-virtual operator construction, direct CPHF, scalar-adjoint contractions, cross-geometry AO overlap, SCF-control transfer, and root snapshots.

`deepks.deephf.capabilities` remains PySCF-neutral and owns force-model, projector-metadata, scalar-output, dtype, shape, finite-value, determinism, and model-state validation.

The method, direct driver, Z-vector driver, force-data producer, and scanner consume adapter result objects instead of reaching into PySCF response internals. PySCF-private molecular arrays and basis/ECP metadata, the RHF Hessian derivative helper, CPHF solve, induced-potential construction, and cross-AO overlap remain behind this module boundary.

The accepted P3B reference is the exact native, undecorated, converged, real, closed-shell molecular `pyscf.scf.hf.RHF` state described in P0, with PySCF 2.14, spherical all-electron point-nucleus AOs, occupations of zero or two, a fixed compatible projector, a differentiable descriptor, and a stable audited occupied-virtual operator.

## 10. Strict fresh-reference scanner

Calling `gradient_driver.as_scanner(root_overlap_tolerance=0.5)` creates `RHFDeePHFGradientScanner` with an immutable backend and immutable copied backend options. It snapshots the method's direct `response_options`, the method's Z-vector `adjoint_options`, and the selected driver's override options independently; later mutation of any originating dictionary does not change the scanner configuration. Both explicit `direct` and `zvector` drivers retain their selected backend through the scanner.

Each call accepts either an exact native hook-free `pyscf.gto.Mole` with the same static molecular fingerprint as the template or a real numerical coordinate array with shape `(n_raw_atom, 3)` interpreted in `Bohr`. Exact Mole coordinates are read through the characterized native class method, and static plus scientific fingerprints are checked before and after the read so an instance method override or in-flight input mutation fails before fresh SCF. An optional `atmlst` is validated before any fresh SCF work.

For every geometry, the scanner deep-copies the molecule template, installs the new coordinates, constructs a new exact native `pyscf.scf.hf.RHF`, restores the validated static SCF controls, disables checkpoint, callback, and external DIIS-file state, and calls `kernel(dm0=None)`. It never warm-starts from a previous density, orbital coefficient matrix, reference object, DeePHF method, descriptor, response, adjoint, or gradient driver.

The static fingerprint binds molecule class and build state, atom, AO, and basis structure, charge, spin, electron count, spherical/cartesian and symmetry controls, nuclear model, basis, ECP and pseudopotential metadata, AO labels, and all noncoordinate PySCF molecular arrays. A molecule with different static metadata is rejected before a reference is built.

Root continuity is measured by the singular values of `C_occ(previous).T S_cross C_occ(candidate)`. Occupations must remain identical and the minimum singular value must meet `root_overlap_tolerance`; this criterion is invariant to occupied-orbital signs and rotations and rejects an occupied-virtual subspace swap.

The scanner fingerprints model structure, semantic module metadata, parameters, buffers, generators, evaluation flags, and `state_dict` content. Every named module must remain in evaluation mode and every local or global module execution-hook registry must be empty before fresh SCF; strict method evaluation additionally proves deterministic complete energy and sensitivity without global RNG consumption. A legal model change between calls causes complete recomputation, while any model change during one energy-and-gradient evaluation is a hard failure.

At the start of every call the scanner clears its public result. It publishes `mol`, `reference`, `method`, `gradient_driver`, `e_tot`, immutable `de`, `converged`, and the model fingerprint atomically only after the complete energy and gradient succeed, and only then advances the root anchor. Any coordinate, static-state, SCF, root, model, descriptor, response, adjoint, or gradient failure leaves no current result, preserves the preceding accepted root anchor, and permits a later valid call to recover without stale state.

Successful `A -> B -> A` geometry sequences build distinct object graphs on all three calls and agree with independently constructed fresh DeePHF methods at the same geometries. The original reference, method state, model values and gradients, and input molecule remain unchanged.

## 11. P3A force-data boundary

`deepks.deephf.generate_rhf_force_frame(...)` explicitly selects `backend="direct"`, requires its `RHFResponse`, and stores the verified model-independent identity `dq_dR_relaxed = dq_dR_explicit + dq_dR_response`. The runnable P3A teacher-data example also selects the direct driver explicitly.

An `RHFAdjoint` contains only the correction-specific scalar contraction needed for inference. It exposes no `density_response`, `first_order_density`, `dq_dR_response`, or `dq_dR_relaxed`, and the Z-vector driver cannot be passed off as a P3A relaxed-Jacobian producer.

A checkpoint produced by the strict P3A force-training workflow remains a valid Z-vector correction model after its force metadata, projector, feature count, dtype, and state have passed the existing checkpoint validation.

## 12. Acceptance coverage and commands

The deterministic P3B suite covers a nonsymmetric generic transpose problem, literal and independently applied transpose residuals, complete-model `W` against AO-density autograd, the bilateral closed-shell `b`, an independent AO-to-MO integral construction of `A`, one solve for one scalar objective, all objective and adjoint metric formulas, direct-versus-Z-vector agreement for every common gradient partition, three-step fresh-reference total-energy finite differences, zero and constant corrections, force sign and atom selection, independent method and driver option namespaces, forbidden direct and coordinate-wise density paths, solver and residual fault injection, operator stability and condition gates, objective and model failures, stochastic and state-mutating model rejection, complete element-constant validation, projector and scientific-state mutation, foreign or forged adjoint data, nondifferentiable descriptors, P3A checkpoint inference, fresh direct and Z-vector scanners, exact Mole input validation, static-state validation, SCF restart semantics, occupied-subspace root tracking, model fingerprints, atomic failure publication, recovery, `A -> B -> A` fresh-method agreement, and architecture checks that isolate the reference-neutral adjoint and all PySCF 2.14 compatibility facilities.

The final P3B verification on Python 3.11 and PySCF 2.14 completed with 153 Z-vector tests, 105 analytic-force tests, 99 force-training tests, 45 baseline tests, and 402 tests in the complete repository suite. Locked dependency synchronization, source and wheel builds, Python bytecode compilation, and `git diff --check` also completed successfully.

Run the complete acceptance sequence from the repository root:

```bash
uv sync --locked --python 3.11
uv run pytest tests/zvector_inference
uv run pytest tests/analytic_forces
uv run pytest tests/force_training
uv run pytest tests/baseline
uv run pytest
uv build
git diff --check
```
