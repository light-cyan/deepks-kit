# P4B RKS DeePHF Direct Oracle

## 1. Status and objective

P4B is implemented and target-accepted for the strict molecular closed-shell pure-LDA RKS domain defined in this document.

The implemented objective is the exact analytic nuclear gradient of the finite-grid energy `e_tot(R) = e_base(R) + e_corr(q(P(R), O(R)))` for one native converged RKS reference, with a complete first-order spin-summed AO density response for every nuclear Cartesian coordinate.

Exactness is relative to the declared PySCF 2.14, LibXC, deterministic atom-centered grid, reference, projector, descriptor, and model contract; both the reference gradient and the CPKS nuclear right-hand side contain the complete grid-coordinate and grid-weight response of that finite-grid energy.

The direct oracle retains occupied-virtual, AO-metric, fixed-grid, XC AO-motion, grid-coordinate, grid-weight, explicit, response, relaxed, native-reference, and correction partitions as independently inspectable runtime values.

The [P0 scientific contract](./p0_scientific_contract.md) remains authoritative for package boundaries, descriptor meaning, derivative terminology, axes, signs, units, differentiability, and hard-failure behavior.

## 2. Implemented package boundary

`deepks.deephf.rks_method.RKSDeePHF` owns perturbative RKS energy composition, complete density and descriptor-response contractions, and construction of the RKS direct-gradient driver.

`deepks.deephf.rks_gradient.RKSDeePHFGradients` owns complete direct-gradient assembly and publishes native-grid, descriptor, AO-metric, occupied-virtual, response, correction, and total partitions.

`deepks.deephf.pyscf_rks.RKSResponseAdapter` owns exact native RKS validation, functional and grid provenance, molecular CPKS, complete AO density reconstruction, residual refinement, physical occupied-virtual operator construction, native grid-response gradient reconstruction, and independent response audits.

Shared projection, spectral descriptor, `dq_dP`, fixed-density explicit derivative, model validation, and model-sensitivity mathematics remain in their established shared packages.

The RKS direct path does not access RHF adjoint, RHF scanner, RHF force-data, or UHF direct objects, and those paths do not consume RKS response state.

## 3. Strict native closed-shell RKS domain

The RKS direct oracle accepts a reference only when every condition below holds.

- The reference has exact type `pyscf.dft.rks.RKS`, is converged, and is attached to an exact molecular `pyscf.gto.mole.Mole` object.
- The molecule has spin zero and an even electron count, is finite, nonperiodic, symmetry-disabled, all-electron, point-nuclear, and real-atom, and uses spherical molecular Gaussian AOs with the full Coulomb interaction.
- The reference and molecule have no active density fitting, solvent, X2C, QM/MM, dispersion, penalty, pseudopotential, ECP, finite-nucleus, range-separated Coulomb, or callable instance-hook state.
- The MO coefficient matrix is complete, square, real, finite, and `numpy.float64`; orbital energies and occupations are complete, real, finite, ordered, and `numpy.float64`.
- Occupations are exactly zero or two, occupy the lowest canonical orbitals, reproduce the molecular electron count, and leave at least one occupied and one virtual orbital.
- The orbitals satisfy AO-metric orthonormality, the density is symmetric with the correct electron count and closed-shell metric idempotency, and the stored orbitals, energies, effective potential, and total energy satisfy the independently reconstructed finite-grid RKS equations.
- The AO overlap is nonsingular, the minimum occupied-virtual gap exceeds `orbital_gap_tolerance`, and the physical CPKS operator passes the configured dimension, symmetry, stability, and condition-number gates.
- The projector metadata, descriptor feature count, scalar correction output, double-precision model state, deterministic complete model sensitivity, and ordered-spectrum differentiability satisfy the shared strict force contract.
- The runtime PySCF version belongs to the `2.14` series characterized by the RKS compatibility adapter.

The method binds the exact reference, molecule, projector, and descriptor state; every accepted response binds the exact reference object, functional, prebuilt grid, orbital state, density, total energy, and response controls; strict gradient assembly separately validates the complete model state and sensitivity before and after contraction.

## 4. Pure-LDA LibXC contract

The accepted functional has normalized LibXC components `((1, 1.0), (7, 1.0))`, corresponding to `LDA_X + LDA_C_VWN` with no hybrid, range-separated, nonlocal-correlation, or custom-functional contribution.

Any LibXC string alias is accepted only when parsing yields those exact normalized component identifiers and coefficients and its evaluated energy density, potential, and second derivative are bitwise identical to the canonical `LDA_X,LDA_C_VWN` LibXC cache on the adapter's audit density sequence.

The numerical integrator has exact type `pyscf.dft.numint.NumInt`, uses the characterized native LibXC `7.0.0` backend with `NumInt.cutoff=1e-13`, has no registered custom functional or callable instance hook, leaves its range-separation parameter unset, reports XC type `LDA`, and supports derivative order two.

`RKSFunctionalProvenance` records the backend module, exact backend version and reference, numerical-integration cutoff, XC type, normalized components, hybrid coefficient, range-separation tuple, NLC state, and canonical evaluated signature.

The ground-state electron count, XC energy, and XC potential are independently rebuilt by dense quadrature on the accepted grid and must match the native `nr_rks` result, while the complete effective potential must match direct molecular `J[P] + V_xc[P]`.

## 5. Deterministic finite-grid and grid-response contract

The accepted grid has exact type `pyscf.dft.gen_grid.Grids`, is bound to the reference molecule, is prebuilt with nonzero-table metadata, and is byte-for-byte reproducible by a fresh deterministic build.

Every element uses the explicit atom-grid specification `(20, 50)`, the grid is unpruned, `alignment=1`, `sort_grids=False`, `symmetry=False`, `small_rho_cutoff=0.0`, and the grid cutoff is exactly `1e-15`.

The radial method is the native Treutler-Ahlrichs method, the atomic-radii adjustment is the native Treutler adjustment, and the partition is the original Becke scheme; the canonical PySCF 2.14 `BRAGG_RADII` value is a finite `numpy.float64` array with shape `(131,)` and dtype-shape-content SHA-256 fingerprint `d5eeefc53bb8261154cd2317ff60e5e642dd9cde1d1f283647b7956756b74a43`.

The cached coordinates, weights, atom indices, quadrature weights, and nonzero table have validated shapes, dtypes, values, and fingerprints and must be exactly equal in dtype, shape, and value to a fresh strict build; AO ordering is independently fingerprinted.

The response generator must retain the characterized `pyscf.grad.rks.grids_response_cc` object identity and return exactly one block per raw host atom in atom order; each host block must cover the nonempty contiguous cached `atm_idx == host_atom` range without repartitioning.

For a host block with `n_host_point` points, coordinates have shape `(n_host_point, 3)`, weights have shape `(n_host_point,)`, and the grid-weight derivative `w1` has shape `(n_raw_atom, 3, n_host_point)`; all three arrays must be finite `numpy.float64`, and block coordinates must exactly equal the corresponding cached energy-grid coordinates.

The characterized PySCF 2.14 response generator can differ from cached energy-grid tail weights by approximately `1e-198`; only block-weight differences satisfying `rtol=0` and `atol=1e-180` are accepted, and every accepted block is normalized to the cached energy-grid weights before response Hamiltonian or native-gradient quadrature.

Concatenated `w1` has shape `(n_raw_atom, 3, n_grid)`, must satisfy nuclear translation invariance with maximum residual at most `1e-10`, and is independently audited against fresh strict-grid central finite differences with displacement `h=1e-5 Bohr`, `rtol=1e-7`, and `atol=1e-6` before any CPKS solve.

`RKSGridProvenance` records the grid class and build generator, qualified response-generator identity, per-element atom grid, radial and partition functions, canonical radii fingerprint, pruning, alignment, cutoffs, sort policy, point count, every cached-grid fingerprint, the concatenated grid-weight-derivative fingerprint, and AO-order fingerprint.

The deterministic grid definition is fixed, while its atom-centered coordinates and Becke partition weights remain geometry-dependent; the complete derivative therefore includes both grid-coordinate and grid-weight response.

The response Hamiltonian is partitioned as `H^R = H_fixed_grid^R + H_xc_grid_coordinate^R + H_xc_grid_weight^R`, where `H_fixed_grid^R` contains the core, Coulomb, and XC AO-motion terms and the adapter independently verifies the XC AO-motion term against PySCF's fixed-grid derivative.

The native reference gradient is partitioned as `g_RKS = g_without_grid_response + g_xc_grid_coordinate + g_xc_grid_weight`; the complete native driver uses `grid_response=True`, and the fixed-grid driver is retained only as an audited partition.

## 6. Closed-shell pure-LDA response equations

Accepted occupied orbitals have `n_i = 2`, the spin-summed ground-state density is `P = C_o diag(n_i) C_o.T`, and a virtual-occupied response amplitude produces `delta P(X) = C_v X (C_o diag(n_i)).T + C_o diag(n_i) X.T C_v.T`.

For a trial density response, the physical pure-LDA induced potential is `G_RKS[delta P] = J[delta P] + K_xc[delta P]`, where `K_xc` is the dense finite-grid contraction of the LibXC LDA kernel `f_xc(rho)` with the induced density.

The unshifted physical occupied-virtual operator is `(A X)[a,i] = (epsilon_a - epsilon_i) X[a,i] + (C_v.T G_RKS[delta P(X)] C_o)[a,i]`.

For coordinate `R`, the occupied-space metric response is `U_oo^R = -0.5 S_oo^R`, its AO density contribution is `P_metric^R = -0.5 P S^R P`, and the occupied-virtual amplitudes solve `A X^R = -B^R` with the complete finite-grid `H^R`, overlap term, and induced potential of `P_metric^R`.

The occupied-virtual density is `P_occupied_virtual^R = C_v X^R (C_o diag(n_i)).T + C_o diag(n_i) (X^R).T C_v.T`, and the complete basis-aware response is `P^R = P_metric^R + P_occupied_virtual^R`.

The independent physical residual is rebuilt from the complete `H^R`, overlap derivative, orbital energies, direct Coulomb response, dense LibXC `f_xc` response, and the returned occupied-virtual amplitudes; successful return from PySCF CPHF is not an acceptance criterion by itself.

The descriptor identities are `dq_dR_response = dq_dP : P^R` and `dq_dR_relaxed = dq_dR_explicit + dq_dR_response`.

## 7. Runtime API and tensor contract

The following RKS classes and functions are exported from `deepks.deephf`: `RKSDeePHF`, `RKSDeePHFGradients`, `RKSFunctionalProvenance`, `RKSGridProvenance`, `RKSNativeGradient`, `RKSResponse`, `RKSResponseAdapter`, `RKSResponseDiagnostics`, `RKSResponseError`, and `validate_rks_reference`.

| API | Result |
|---|---|
| `validate_rks_reference(reference)` | The same exact native RKS reference after the complete strict capability, LibXC, grid, AO-state, and energy audit. |
| `RKSDeePHF(reference, model, projector_basis=..., response_options=...)` | A perturbative energy method bound to one strict native closed-shell RKS state. |
| `RKSDeePHF.kernel()` | Scalar `e_tot = e_base + e_corr` without entering CPKS, while retaining `e_base`, `e_corr`, and `e_tot`. |
| `RKSDeePHF.ao_density()` | Native spin-summed `P` with shape `(n_ao, n_ao)`. |
| `RKSDeePHF.response(**options)` | A newly solved audited immutable `RKSResponse` for every nuclear Cartesian coordinate. |
| `RKSDeePHF.first_order_density(response=None, **options)` | Complete `P^R` from a fresh response or from any retained trusted response produced by the same method and reaudited by its originating adapter. |
| `RKSDeePHF.dq_dR_explicit()` | Fixed-`P` descriptor motion. |
| `RKSDeePHF.dq_dR_response(response=None, **options)` | Descriptor contribution from the complete RKS density response. |
| `RKSDeePHF.dq_dR_relaxed(response=None, **options)` | Complete relaxed descriptor derivative. |
| `RKSDeePHF.nuc_grad_method(backend="direct", **options)` | A strict `RKSDeePHFGradients` direct driver. |
| `RKSDeePHF.gradient(...)`, `RKSDeePHF.forces(...)` | Complete analytic gradient and its exact negative. |
| `RKSDeePHFGradients.kernel(atmlst=None)` | Complete gradient for all raw atoms or selected raw-atom indices. |
| `RKSDeePHFGradients.run(atmlst=None)`, `forces(atmlst=None)` | Populated driver or negative gradient under the same selection contract. |
| `RKSResponseAdapter.linear_response_problem()` | The audited physical RKS operator through the reference-neutral `ScalarAdjointProblem` protocol. |
| `RKSResponseAdapter.audit_response_equations(response)` | Full independent reconstruction and equation audit of one immutable adapter response. |

One `RKSDeePHF` method retains at most eight successful response objects that it produced, together with each originating adapter and original integrity fingerprint; any retained response remains reusable after later solves, while a foreign, changed, or evicted response is rejected.

| Value | Runtime axes | Unit |
|---|---|---|
| `mo_response` and its metric and occupied-virtual partitions | `(n_raw_atom, 3, n_mo, n_occupied)` | `Bohr^-1`. |
| `coefficient_response` and its partitions | `(n_raw_atom, 3, n_ao, n_occupied)` | `Bohr^-1`. |
| `density_response` and its partitions | `(n_raw_atom, 3, n_ao, n_ao)` | `Bohr^-1` in the numerical AO representation. |
| `overlap_derivative` | `(n_raw_atom, 3, n_ao, n_ao)` | `Bohr^-1`. |
| Hamiltonian derivative and fixed-grid, AO-motion, grid-coordinate, and grid-weight partitions | `(n_raw_atom, 3, n_ao, n_ao)` | `Eh/Bohr`. |
| `orbital_response_residual` | `(n_raw_atom, 3, n_virtual, n_occupied)` | `Eh/Bohr`. |
| `dq_dR_explicit`, `dq_dR_response`, `dq_dR_relaxed` | `(n_raw_atom, 3, n_descriptor_atom, n_feature)` | `Bohr^-1`. |
| Native-grid, correction, and total gradients | `(n_raw_atom, 3)` before optional atom selection | `Eh/Bohr`. |

Every strict response array is real, finite, immutable, C-contiguous `numpy.float64` and is covered by the response integrity fingerprint.

## 8. Gradient assembly and retained partitions

The explicit correction gradient is `g_explicit^R = sum_I,k (partial e_corr / partial q[I,k]) dq_dR_explicit[R,I,k]`.

The density-response contraction is retained as `g_response = g_metric + g_occupied_virtual`, where each part contracts the common correction AO objective `W = partial e_corr / partial P` with the corresponding AO density response.

The correction and complete gradients satisfy `g_corr = g_explicit + g_response` and `g_tot = g_RKS_native + g_corr`.

`RKSDeePHFGradients` publishes `reference_gradient`, `reference_gradient_without_grid_response`, `reference_gradient_xc_grid_coordinate`, `reference_gradient_xc_grid_weight`, the native reconstruction residual, `dq_dR_explicit`, `dq_dR_response`, `dq_dR_relaxed`, `correction_gradient_explicit`, `correction_gradient_metric`, `correction_gradient_occupied_virtual`, `correction_gradient_response`, `correction_gradient`, `de_full`, and the selected `de`.

A missing correction, a zero correction, and a coordinate-independent constant correction have zero model sensitivity and reduce the analytic gradient to the complete native grid-response RKS gradient while retaining the corresponding total-energy value or constant offset.

## 9. Solver controls, diagnostics, and invariants

| Option | Default | Acceptance meaning |
|---|---:|---|
| `cphf_tolerance` | `1e-11` | Internal tolerance passed to the PySCF CPHF solve and every residual-refinement solve. |
| `residual_tolerance` | `1e-9` | Maximum accepted independently reconstructed physical occupied-virtual residual. |
| `invariant_tolerance` | `1e-9` | Maximum accepted reconstruction, metric, idempotency, particle-number, and translation residual. |
| `orbital_gap_tolerance` | `1e-7` | Strict lower bound for the occupied-virtual gap in `Eh`. |
| `max_cycle` | `100` | Maximum iterations passed to each PySCF CPHF solve. |
| `max_refinement_cycles` | `3` | Maximum normalized physical-residual correction solves after the initial CPHF solve. |
| `level_shift` | `0.0` | Solver level shift; the operator audit always examines the unshifted physical operator. |
| `operator_stability_tolerance` | `1e-6` | Strict lower bound for the minimum eigenvalue of the physical CPKS operator in `Eh`. |
| `operator_condition_tolerance` | `1e8` | Maximum accepted spectral condition number. |
| `operator_symmetry_tolerance` | `1e-10` | Maximum accepted absolute matrix symmetry residual. |
| `operator_dimension_limit` | `512` | Maximum occupied-virtual dimension admitted to the explicit dense operator audit. |

`RKSResponseDiagnostics` records the orbital gap; PySCF and LibXC versions; normalized functional components; grid point count and fingerprints; finite-grid electron count; residual maximum, RMS, tolerance, and refinement history; every active solver and operator control; operator spectral bounds, condition, and symmetry; induced-potential, fixed-grid XC, Hamiltonian, and density reconstruction residuals; occupied-space metric, first-order idempotency, particle-number, and translation residuals.

The independent first-order invariants are `P^R S P + P S^R P + P S P^R - 2 P^R = 0` and `Tr(P^R S) + Tr(P S^R) = 0` for the accepted spin-summed closed-shell density convention.

The adapter also checks the symmetric occupied-space gauge `U_oo^R = -0.5 S_oo^R`, every MO, coefficient, density, Hamiltonian, functional, and grid partition reconstruction, and the translational sum of the complete first-order density.

## 10. PySCF 2.14 and LibXC compatibility boundary

All RKS-specific access to PySCF private or semi-private response, DFT, grid, and molecular state is isolated in `deepks.deephf.pyscf_rks`.

That module is the unique owner of `pyscf.hessian.rks.Hessian.make_h1`, `_get_vxc_deriv1`, `pyscf.scf.cphf.solve`, direct molecular J construction, `NumInt` LDA and `f_xc` evaluation, LibXC cache and custom-functional inspection, `pyscf.grad.rks.grids_response_cc`, grid and radii controls, native RKS gradient construction, and the private arrays used by scientific-state fingerprints.

The adapter converts PySCF-specific values into immutable `RKSResponse`, frozen functional and grid provenance, frozen response diagnostics, and immutable `RKSNativeGradient` records before method or gradient code consumes them.

Architecture tests bind each PySCF and LibXC facility to the RKS adapter, prohibit private PySCF state in the RKS method and gradient modules, enforce unique ownership of exported RKS symbols, and preserve isolation from the RHF and UHF object graphs.

## 11. Failure behavior, state safety, and API boundaries

| Boundary | Explicit behavior |
|---|---|
| Reference type, convergence, molecular physics, decorations, hooks, orbitals, occupations, canonical state, AO overlap, effective potential, finite-grid quadrature, energy reconstruction, PySCF series, orbital gap, operator dimension, stability, or conditioning | `DeePHFCapabilityError` is raised before a response or force result is returned. |
| LibXC version, components, parameter signature, backend, `NumInt` cutoff, derivative order, hybrid, range separation, NLC, or custom-functional state | `DeePHFCapabilityError` is raised during reference validation. |
| Grid class, ownership, atom-grid counts, radial method, canonical radii content, partition, pruning, alignment, cutoffs, cached arrays, fresh-build equality, response-generator identity, host-block boundaries, response arrays, cached-weight tolerance, `w1` translation, independent `w1` finite differences, or grid hooks | `DeePHFCapabilityError` is raised during reference validation. |
| Projector metadata, model dtype, state, output, sensitivity, determinism, or descriptor differentiability | The shared capability or descriptor error propagates before CPKS. |
| PySCF derivative construction, grid-response quadrature, CPHF solve or refinement, nonfinite response, operator asymmetry, excessive physical residual, invariant failure, or reconstruction failure | `RKSResponseError` is raised without a density, descriptor, or gradient fallback. |
| Supplied response with foreign method identity, unavailable trusted adapter, changed integrity, stale science state, mutable or invalid arrays, inconsistent provenance, forged diagnostics, or failed equation audit | `RKSResponseError` is raised before response consumption. |
| Invalid response-control scalar, tolerance, cycle limit, operator dimension, or unknown direct option | `ValueError` is raised during adapter construction or backend-option validation. |
| A supplied response combined with response-option keywords | `ValueError` is raised before either input can be ignored. |
| A backend other than `direct`, nonempty RKS adjoint options, or an adjoint request | `DeePHFCapabilityError` is raised at the selected RKS boundary. |
| RKS gradient-scanner construction | `RKSResponseError` is raised. |
| Use of an RKS reference with the RHF force-data producer or RHF gradient and adjoint drivers | The exact RHF type boundary rejects the request. |
| Invalid atom selection or corrupted method-driver binding | The selection or binding error is raised before response publication. |

`RKSDeePHF.response` publishes no new trusted result until a solve succeeds; success adds the response, its originating adapter, and its original integrity to the method's bounded retention set without invalidating previously retained responses, and the oldest response ceases to be trusted when the eight-response limit is exceeded.

`RKSDeePHFGradients.kernel` clears every public result before evaluation and clears them again on any failure, so an unsuccessful call cannot expose a preceding native gradient, response, descriptor partition, correction partition, or total gradient.

The RKS path never substitutes `dq_dR_explicit`, an RHF or UHF density response, a scalar adjoint, a fixed-grid native gradient, or a grid-response-omitting nuclear right-hand side for the complete accepted RKS direct result.

## 12. Deterministic acceptance and commands

The numerical oracle uses a symmetry-disabled, spherical, distorted neutral `H2O/STO-3G` singlet with fixed coordinates in Bohr, `LDA_X + LDA_C_VWN`, a prebuilt unpruned `(20, 50)` grid on every atom, projector shells `[[0, [0.8, 1.0]], [1, [0.3, 1.0]]]`, and a deterministic nontrivial double-precision nonlinear correction model.

The accepted central reference has `3000` grid points, `7` AOs, a `10`-dimensional occupied-virtual response operator, a minimum orbital gap above `0.36 Eh`, a minimum descriptor gap above `5e-2`, and an operator condition number below `44`; every displaced occupied-subspace overlap remains above `0.99`.

Every central displacement independently converges a fresh native RKS reference with `conv_tol=1e-13`, `conv_tol_grad=1e-10`, `conv_tol_cpscf=1e-12`, `max_cycle=100`, and the exact functional and grid contract; the sequence preserves occupations, AO labels, grid size, grid ordering, stable root, positive gap, and differentiable descriptor.

Central differences at `(1e-3, 3e-4, 1e-4) Bohr` independently validate the complete AO density derivative, relaxed descriptor derivative, native finite-grid gradient, and complete `e_base + e_corr` gradient.

Independent dense AO and grid oracles validate the Coulomb and LibXC `f_xc` operator blocks, operator spectrum, overlap derivative, fixed-grid XC AO motion, grid-coordinate and grid-weight Hamiltonian terms, native grid-response gradient partitions, metric density, complete physical residual, and fresh-grid fixed-density Fock differences.

The grid acceptance suite independently validates generator identity, host-atom block boundaries, cached energy-grid normalization, full `w1` shape, dtype, finiteness, translation, and `h=1e-5 Bohr` finite differences, and detects block repartitioning, response-weight corruption, radii-table mutation, and a translation-preserving `w1` fault before CPKS.

Cross-molecule smoke checks independently converge and validate strict `H2/STO-3G` and `LiH/STO-3G` references under the same LibXC and grid contract; the `H2` case additionally completes and reaudits the full density response.

The acceptance suite also verifies omission detectability for Coulomb, `f_xc`, metric, AO motion, all grid response, grid-coordinate response, and grid-weight response; translation and nonorthogonal first-order invariants; immutable retained-response audits and reuse across later solves; solver, quadrature, residual, operator, provenance, model, projector, and differentiability failures; result clearing; atom selection; force sign; zero, constant, and absent corrections; and isolation from RHF and UHF facilities.

Run the accepted verification sequence from the repository root:

```bash
uv sync --locked --python 3.11
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
