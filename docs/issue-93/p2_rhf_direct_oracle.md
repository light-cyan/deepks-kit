# P2 RHF DeePHF Direct Oracle

## 1. Status and objective

P2 implementation is in progress, and its exit gate remains open until the response, finite-difference, regression, and documentation evidence in this document passes together.

The P2 objective is the exact analytic nuclear gradient of the perturbative energy `e_tot(R) = e_base(R) + e_corr(q(P(R), O(R)))` for the strict molecular RHF support domain defined below.

The direct oracle evaluates the first-order reference response for every nuclear Cartesian coordinate, retains the occupied-virtual and AO-metric contributions to the numerical AO density derivative, forms the complete relaxed descriptor derivative, and contracts it with the correction-model sensitivity.

The [P0 scientific contract](./p0_scientific_contract.md) remains authoritative for method meaning, canonical names, axes, signs, units, degeneracy semantics, and failure boundaries.

## 2. Implemented package boundary

`deepks.deephf.method.DeePHF` owns perturbative energy composition, descriptor-response contraction, and construction of the analytic-gradient driver.

`deepks.deephf.gradient.RHFDeePHFGradients` owns the direct-oracle gradient assembly and retains the explicit, response, relaxed, reference, and correction partitions as separate runtime values.

`deepks.deephf.pyscf_rhf.RHFResponseAdapter` owns the molecular RHF nuclear CPHF solve, complete first-order AO density reconstruction, residual refinement, and independent response audits.

`deepks.deephf.capabilities` owns strict reference, model, projector-metadata, scalar-output, dtype, shape, and finite-value validation.

Shared projection, descriptor, `dq_dP`, and fixed-density explicit derivative mathematics remain in `deepks.descriptor`; the DeePHF package does not import `deepks.deepks`.

## 3. Strict RHF support domain

The direct oracle accepts an object only when every condition in this section is satisfied before the response result is used.

### 3.1 Reference state

- The reference has exact type `pyscf.scf.hf.RHF`, is converged, and is attached to an exact molecular `pyscf.gto.mole.Mole` object.
- The molecule has spin zero, finite coordinates, spherical Gaussian AOs, point nuclei, the full Coulomb interaction, no molecular symmetry, no ECP or pseudopotential, and no zero-charge ghost centers.
- Density fitting, solvent, X2C, QM/MM, dispersion, penalty state, callable instance-level reference overrides, and callable instance-level molecule overrides are rejected.
- Orbital coefficients, orbital energies, occupations, AO overlap, core Hamiltonian, effective potential, and AO density are real, finite, and shape-compatible with the molecular AO dimension.
- The MO coefficient matrix is complete and square with `n_mo = n_ao`; occupations are exactly two for the lowest `N/2` orbitals and zero otherwise, their sum equals the molecular electron count, and both occupied and virtual spaces exist.
- The orbitals satisfy `C^T S C = I` within `1e-8`, the AO overlap minimum eigenvalue exceeds `1e-10`, `Tr(P S)` agrees with the molecular electron count within `1e-8`, and the canonical residual `F C - S C epsilon` does not exceed `1e-7`.
- The stored RHF total energy agrees within `1e-8 Eh` with the value recomputed from the accepted AO density, core Hamiltonian, Fock matrix, and nuclear repulsion.
- The minimum occupied-virtual orbital-energy gap is finite and exceeds the configured `orbital_gap_tolerance`, whose default is `1e-7 Eh`.
- The complete unshifted singlet occupied-virtual response operator is explicitly constructed inside the configured dimension limit, is symmetric within the configured tolerance, has a strictly positive minimum eigenvalue, and satisfies the configured condition-number bound.
- The runtime PySCF version belongs to the `2.14` series characterized by the adapter.

### 3.2 Correction model and projector

- The correction is either `None`, which denotes zero correction, a `torch.nn.Module`, or a `CorrNet` checkpoint path loaded by `DeePHF`.
- Every floating model parameter and buffer is finite `torch.float64`, and complex model state is rejected.
- A declared model `input_dim` equals the descriptor feature count, and declared model projector metadata `_pbas` equals the normalized projector basis used by the descriptor.
- Model evaluation produces exactly one real, finite `torch.float64` tensor element, and any elemental constant is scalar, real, and finite.
- The complete model sensitivity `partial e_corr / partial q`, including preprocessing and every model branch, is finite.
- Ordered descriptor blocks pass the active differentiability validator: isolated eigenvalues satisfy the scale-aware gap test, accepted structural zero blocks satisfy the rank-based zero test, and model sensitivities are equal within an accepted repeated zero subspace.

The runtime differentiability validator currently audits the central reference state; P2 numerical acceptance separately requires stable descriptor behavior across the documented displaced-reference sequence before the exit gate can close.

## 4. Response equations and partitions

Let `A` index raw atoms, `x` index Cartesian coordinates, `mu` and `nu` index AOs, `p` index all MOs, `i` index occupied MOs, `a` index virtual MOs, and `I,k` index descriptor atoms and features.

The adapter represents the first-order occupied-orbital coefficients as `C^A_mu i = sum_p C_mu p U^A_p i`, where `U` has runtime axes `(n_raw_atom, 3, n_mo, n_occ)`.

The occupied-occupied block supplies the AO-metric response and satisfies `U^A_ji + U^A_ij + S^A_ij = 0`, where `S^A_ij = C_i^T S^A C_j`.

The occupied-virtual block is obtained from the molecular RHF CPHF equation, and the independently evaluated virtual-occupied residual is `r^A_ai = h^A_ai + V^A_ai - S^A_ai epsilon_i + (epsilon_a - epsilon_i) U^A_ai`.

For an occupied-virtual trial amplitude `X_ai`, the physical unshifted response operator is `A[X]_ai = (epsilon_a - epsilon_i) X_ai + C_a^T (J[D[X]] - 0.5 K[D[X]]) C_i`, where `D[X] = 2 [(C_a X) C_i^T + C_i (C_a X)^T]`; the stability and conditioning audit always uses this operator, so the solver `level_shift` cannot mask a singular or unstable reference.

For accepted closed-shell occupations `n_i = 2`, the numerical AO density response is `P^A_mu nu = sum_i n_i [(C U^A)_mu i C_nu i + C_mu i (C U^A)_nu i]`.

The response object retains `U_metric`, `U_occupied_virtual`, `P_metric`, and `P_occupied_virtual`, and it enforces `P^A = P^A_metric + P^A_occupied_virtual` by construction.

The fixed-density explicit descriptor derivative is `dq_dR_explicit`, the density-response contribution is `dq_dR_response[A,x,I,k] = sum_mu,nu dq_dP[I,k,mu,nu] P^A[mu,nu]`, and the complete derivative is `dq_dR_relaxed = dq_dR_explicit + dq_dR_response`.

The correction gradient is `g_corr[A,x] = sum_I,k (partial e_corr / partial q[I,k]) dq_dR_relaxed[A,x,I,k]`, the reported energy gradient is `g_tot = g_RHF + g_corr`, and the nuclear force is `f_tot = -g_tot`.

The gradient driver retains `correction_gradient_explicit` and `correction_gradient_response` in addition to their sum so the response contribution cannot be confused with the fixed-density term.

## 5. Runtime tensor axes and units

| Value | Runtime axes | Unit |
|---|---|---|
| `mo_response`, `mo_response_metric`, `mo_response_occupied_virtual` | `(n_raw_atom, 3, n_mo, n_occ)` | `Bohr^-1` |
| `coefficient_response`, `coefficient_response_metric`, `coefficient_response_occupied_virtual` | `(n_raw_atom, 3, n_ao, n_occ)` | `Bohr^-1` |
| `density_response`, `density_response_metric`, `density_response_occupied_virtual` | `(n_raw_atom, 3, n_ao, n_ao)` | `Bohr^-1` in the numerical AO representation |
| `overlap_derivative`, `hamiltonian_derivative` | `(n_raw_atom, 3, n_ao, n_ao)` | `Bohr^-1` and `Eh/Bohr`, respectively |
| `orbital_response_residual` | `(n_raw_atom, 3, n_virtual, n_occ)` | `Eh/Bohr` |
| `dq_dP` | `(n_descriptor_atom, n_feature, n_ao, n_ao)` | Dimensionless with respect to the numerical AO density |
| `dq_dR_explicit`, `dq_dR_response`, `dq_dR_relaxed` | `(n_raw_atom, 3, n_descriptor_atom, n_feature)` | `Bohr^-1` |
| `reference_gradient`, correction-gradient partitions, `de_full`, `de` | `(n_raw_atom, 3)` before optional atom selection | `Eh/Bohr` |

All strict scientific arrays use real double precision.

## 6. PySCF 2.14 adapter boundary

All direct-oracle use of the low-level `pyscf.scf.cphf.solve` interface and the semi-private RHF Hessian `make_h1` helper is isolated in `deepks/deephf/pyscf_rhf.py`.

The adapter checks the PySCF major-minor series before solving, converts PySCF-specific return shapes into the stable `RHFResponse` dataclass, makes every array immutable, and adds state and response-integrity fingerprints before exposing the result to the method and gradient layers.

The adapter constructs `S^A` in the numerical AO basis, obtains the complete first-order RHF Hamiltonian through `pyscf.hessian.rhf.Hessian(reference).make_h1`, and evaluates the induced RHF potential as `J[P^A] - 0.5 K[P^A]`.

No PySCF response helper is imported into `deepks.descriptor`, and method-neutral descriptor code does not own reference-response semantics.

## 7. Solver, refinement, and independent audits

The response controls accepted by `RHFResponseAdapter` and by `DeePHF(response_options=...)` are:

| Option | Default | Meaning |
|---|---:|---|
| `cphf_tolerance` | `1e-11` | Internal tolerance passed to each PySCF CPHF solve. |
| `residual_tolerance` | `1e-9` | Maximum accepted independently recomputed occupied-virtual residual. |
| `invariant_tolerance` | `1e-9` | Maximum accepted metric, first-order idempotency, and particle-number residual. |
| `orbital_gap_tolerance` | `1e-7` | Minimum accepted occupied-virtual gap in `Eh`. |
| `max_cycle` | `100` | Maximum iterations passed to each PySCF CPHF solve. |
| `max_refinement_cycles` | `3` | Maximum independent residual-correction solves after the initial response. |
| `level_shift` | `0.0` | Level shift passed to PySCF CPHF. |
| `operator_stability_tolerance` | `1e-6` | Strict lower bound for the minimum eigenvalue of the physical response operator in `Eh`. |
| `operator_condition_tolerance` | `1e8` | Maximum accepted spectral condition number of the positive response operator. |
| `operator_symmetry_tolerance` | `1e-10` | Maximum accepted absolute matrix symmetry residual. |
| `operator_dimension_limit` | `512` | Maximum occupied-virtual dimension accepted by the explicit dense condition audit. |

After the initial PySCF solve, the adapter recomputes the occupied-virtual equation residual from `U`, `h^A`, `S^A`, orbital energies, and a newly evaluated induced potential instead of trusting the low-level solver return as convergence evidence.

When the maximum residual exceeds `residual_tolerance`, each active nuclear perturbation is normalized by its residual norm, a correction equation with no metric right-hand side is solved, the scaled correction is added to the virtual-occupied response, and the residual is recomputed; this repeats up to `max_refinement_cycles`.

The adapter independently checks the occupied-space metric identity, the derivative of closed-shell AO-metric idempotency `P S P = 2 P`, and the derivative of particle number `Tr(P S) = N`.

Before solving any nuclear perturbation, the adapter explicitly applies the unshifted physical response operator to the occupied-virtual basis, rejects dimensions above `operator_dimension_limit`, audits matrix symmetry, diagonalizes the symmetrized matrix, rejects nonpositive or insufficiently positive eigenvalues, and rejects excessive condition numbers.

`RHFResponseDiagnostics` records the PySCF version, every solver and operator-audit control, occupied-virtual dimension, response-operator extremal eigenvalues, condition number, symmetry residual, minimum orbital gap, maximum and RMS orbital residuals, metric residual, idempotency residual, particle-number residual, number of refinement cycles, and maximum-residual history from the initial solve through every refinement.

All response arrays are checked for finite values before a result is returned.

## 8. Public P2 API

| API | Result |
|---|---|
| `DeePHF.response(**options)` | A validated `RHFResponse` for every nuclear Cartesian coordinate. |
| `DeePHF.first_order_density(response=None, **options)` | Complete numerical AO density derivative `P^A` after validating any supplied response. |
| `DeePHF.dq_dR_response(response=None, **options)` | Density-response descriptor derivative after validating any supplied response. |
| `DeePHF.dq_dR_relaxed(response=None, **options)` | Sum of explicit and validated response descriptor derivatives. |
| `DeePHF.nuc_grad_method(**options)` | An `RHFDeePHFGradients` driver. |
| `DeePHF.gradient(**options)` | Complete analytic energy gradient. |
| `DeePHF.forces(**options)` | Negative complete analytic energy gradient. |
| `RHFDeePHFGradients.kernel(atmlst=None)` | Complete gradient for all atoms or the selected raw-atom indices. |
| `RHFDeePHFGradients.run(atmlst=None)` | Evaluates the gradient and returns the populated driver. |
| `RHFDeePHFGradients.forces(atmlst=None)` | Negative gradient for all atoms or the selected raw-atom indices. |

`RHFResponse`, `RHFResponseDiagnostics`, `RHFResponseError`, `RHFResponseAdapter`, and `RHFDeePHFGradients` are exported from `deepks.deephf`.

Each `RHFResponse` records the Python identity and a SHA-256 fingerprint of the response-defining molecular and converged RHF state, plus a second SHA-256 integrity fingerprint covering every response field except the integrity digest itself.

Before consuming a supplied `RHFResponse`, `DeePHF` revalidates the current reference, exact response and diagnostic types, reference identity, state and integrity fingerprints, field shapes, real finite double-precision values, array immutability, MO/coefficient/density component sums, occupied-virtual and metric MO support, independent `U -> C^A -> P^A` reconstruction at each response level, reconstructed overlap and Hamiltonian derivatives, the physical response operator, CPHF residual, metric identity, first-order idempotency, particle number, solver-control diagnostics, and recorded thresholds.

The optional `response` argument and response-control keyword arguments are mutually exclusive in APIs that accept both, so a validated stored response cannot silently ignore a second set of solver controls.

Gradient atom selections accept only iterable integer indices, including NumPy integer scalars; boolean, floating, string, negative, and out-of-range indices fail before response work begins.

## 9. Failure behavior

| Category | Exception and behavior |
|---|---|
| Unsupported reference, model, projector metadata, PySCF series, occupation root, orbital gap, response dimension, unstable response operator, or ill-conditioned response operator | `DeePHFCapabilityError` is raised before a force result is returned. |
| Nonfinite descriptor values or sensitivity, incompatible repeated subspace, or an unresolved descriptor gap | `DescriptorDifferentiabilityError` is raised during force-compatibility validation. |
| Hamiltonian-derivative construction failure, CPHF failure, refinement failure, incomplete PySCF result, nonfinite response, excessive independent residual, or invariant violation | `RHFResponseError` is raised and propagated by the method and gradient APIs. |
| Supplied response with a foreign or stale reference state, failed integrity digest, invalid type or shape, mutable or nonfinite arrays, inconsistent partitions or reconstruction, invalid controls or residual diagnostic, or failed recorded threshold | `RHFResponseError` is raised before descriptor contraction. |
| Nonfinite or nonpositive tolerances, nonfinite level shift, or invalid cycle limits | `ValueError` is raised while constructing the response adapter. |
| Simultaneous supplied response and response-control keywords | `ValueError` is raised before either input can be ignored. |
| Noninteger atom selection | `TypeError` is raised by the gradient driver before response work begins. |
| Negative or out-of-range integer atom selection | `IndexError` is raised by the gradient driver before response work begins. |
| Nonfinite assembled total gradient | `RHFResponseError` is raised by the gradient driver. |

The analytic-gradient path has no exception handler that replaces a missing or failed response with `dq_dR_explicit`; reference, response, and correction contractions execute only after strict validation succeeds.

## 10. Deterministic numerical acceptance

P2 acceptance requires deterministic, real, double-precision, low-cost molecular fixtures with independently converged RHF references at every displaced geometry.

- The first-order numerical AO density must agree with central finite differences of independently converged displaced-reference AO densities over a documented step-size sequence, using stable AO ordering and a density-matrix comparison that is invariant to occupied-orbital gauge rotations.
- The occupied-virtual and metric density partitions must sum to the complete first-order density, and omitting the metric partition must be detectably inconsistent with the displaced-reference density and response invariants.
- `dq_dR_response` must equal the contraction of `dq_dP` with the accepted complete first-order density, and `dq_dR_relaxed` must equal both `dq_dR_explicit + dq_dR_response` and the central finite difference of descriptors from independently converged displaced references.
- The complete analytic gradient must agree with central finite differences of `e_base + e_corr` over the accepted step-size sequence.
- A zero correction and a nonzero constant correction must both reduce the analytic gradient to the native RHF gradient within the declared tolerance.
- Deliberately insufficient response convergence, excessive independently recomputed residual, nonfinite response data, invalid model output, incompatible projector metadata, and descriptor nondifferentiability must each produce their documented exception without a fixed-density fallback.
- A reference with a finite HOMO-LUMO gap but a singular complete occupied-virtual response operator must fail the stability gate before a density response or gradient is returned.
- Response and gradient evaluation must leave the converged native RHF energy, density, orbitals, occupations, Fock matrix, and convergence state unchanged.

### 10.1 Direct-oracle fixture and tolerances

The direct finite-difference oracle uses a symmetry-disabled, spherical, distorted `H2O/STO-3G` molecule with coordinates `O (0.13, -0.21, 0.07)`, `H (1.51, 0.12, -0.19)`, and `H (-0.46, 1.62, 0.31)` in Bohr, projector shells `[[0, [0.8, 1.0]], [1, [0.3, 1.0]]]`, and a deterministic four-feature linear correction path.

Every central displacement independently converges RHF with `conv_tol=1e-13`, `conv_tol_grad=1e-10`, `conv_tol_cpscf=1e-12`, and `max_cycle=100`; it preserves occupations and AO labels, retains an occupied-virtual gap greater than `0.8 Eh`, and retains a tested descriptor eigenvalue gap greater than `4e-2`.

The common central-difference steps are `(1e-3, 3e-4, 1e-4) Bohr`.

| Comparison | Relative tolerance | Absolute tolerances for `(1e-3, 3e-4, 1e-4) Bohr` |
|---|---:|---|
| Complete numerical AO density response | `2e-6` | `(5e-7, 1e-7, 1e-7) Bohr^-1` |
| Complete relaxed descriptor response | `2e-6` | `(6e-7, 1e-7, 1e-7) Bohr^-1` |
| Complete `e_base + e_corr` gradient | `3e-6` | `(3e-6, 4e-7, 1e-7) Eh/Bohr` |

The algebraic response and relaxed-descriptor partitions use `rtol=2e-13` and `atol=2e-13`, the density partition uses `rtol=0` and `atol=2e-15`, and the zero and `0.051 Eh` constant corrections reproduce the native RHF gradient with `rtol=2e-12` and `atol=2e-12 Eh/Bohr`.

The accepted fixture reports maximum, RMS, metric, idempotency, and particle-number response diagnostics below `1e-10`; the overlap derivative agrees with its `1e-4 Bohr` central difference using `rtol=2e-8` and `atol=3e-9 Bohr^-1`.

The no-fallback check adds `1e-4` to one returned virtual-occupied CPHF amplitude, disables refinement with `max_refinement_cycles=0`, and requires `RHFResponseError` before any relaxed descriptor or total gradient is stored.

The operator-stability failure fixture uses symmetry-disabled `C2/STO-3G` at a bond length of `3.0 Bohr`: its occupied-virtual gap exceeds `0.3 Eh`, while the complete response operator has a numerical zero mode and is rejected before gradient assembly.

### 10.2 Verification commands

The targeted and repository-wide verification commands are:

```bash
uv run pytest tests/analytic_forces/test_rhf_density_response.py
uv run pytest tests/analytic_forces/test_rhf_relaxed_descriptor.py
uv run pytest tests/analytic_forces/test_deephf_rhf_gradient.py
uv run pytest tests/analytic_forces/test_rhf_response_residual.py
uv run pytest tests/analytic_forces/test_deephf_strict_contract.py
uv run pytest tests/analytic_forces
uv run pytest tests/baseline
uv run pytest
uv build
git diff --check
```

The P2 exit gate closes only when the analytic-force tests contain the density-response, relaxed-descriptor, complete-energy-gradient, zero-correction, constant-correction, residual-failure, invariant, capability, and no-fallback checks described above and every command succeeds.
