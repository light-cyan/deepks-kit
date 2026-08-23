# P4A UHF DeePHF Direct Oracle

## 1. Status and objective

P4A is implemented and target-accepted for the strict molecular UHF domain defined in this document.

The implemented objective is the exact analytic nuclear gradient of `e_tot(R) = e_base(R) + e_corr(q(P_alpha(R) + P_beta(R), O(R)))` for one native converged UHF reference, with a complete coupled alpha/beta first-order AO density response for every nuclear Cartesian coordinate.

The direct oracle retains alpha, beta, occupied-virtual, AO-metric, explicit, response, relaxed, native-reference, and correction partitions as independently inspectable runtime values while preserving the canonical spin-summed descriptor and total energy.

The [P0 scientific contract](./p0_scientific_contract.md) remains authoritative for package boundaries, descriptor meaning, derivative terminology, axes, signs, units, differentiability, and hard-failure behavior.

## 2. Implemented package boundary

`deepks.deephf.uhf_method.UHFDeePHF` owns perturbative UHF energy composition, spin-resolved density accessors, descriptor-response contractions, and construction of the UHF direct-gradient driver.

`deepks.deephf.uhf_gradient.UHFDeePHFGradients` owns complete direct-gradient assembly and publishes spin-resolved and spin-summed explicit, metric, occupied-virtual, response, correction, and total partitions.

`deepks.deephf.pyscf_uhf.UHFResponseAdapter` owns exact native UHF validation, the coupled molecular UC-PHF solve, complete alpha/beta AO density reconstruction, residual refinement, physical occupied-virtual operator construction, and independent response audits.

Shared projection, spectral descriptor, `dq_dP`, fixed-density explicit derivative, and additive explicit-component mathematics remain in `deepks.descriptor`; `deepks.deephf` and `deepks.deepks` remain independent packages.

The RHF method, direct response, scalar adjoint, scanner, and force-data implementations remain in their existing RHF modules and do not own UHF response state.

## 3. Strict UHF support domain

The UHF direct oracle accepts a reference only when every condition below holds.

- The reference has exact type `pyscf.scf.uhf.UHF`, is converged, and is attached to an exact molecular `pyscf.gto.mole.Mole` object.
- The molecule is finite, nonperiodic, symmetry-disabled, all-electron, point-nuclear, and real-atom; it uses spherical molecular Gaussian AOs and the full Coulomb interaction.
- The reference and molecule have no active density-fitting, solvent, X2C, QM/MM, dispersion, penalty, or callable instance-hook state.
- Alpha and beta MO coefficient matrices are complete square real `numpy.float64` arrays, and both orbital-energy and occupation arrays are complete, real, finite, and `numpy.float64`.
- Each spin occupation is exactly zero or one, matches `mol.nelec`, the total electron count, and `mol.spin`, occupies the lowest ordered canonical orbitals, and leaves at least one occupied and one virtual orbital in each spin channel.
- Each spin satisfies `C_sigma.T S C_sigma = I`, AO-density symmetry, `Tr(P_sigma S) = N_sigma`, metric idempotency `P_sigma S P_sigma = P_sigma`, and the canonical UHF equations within the validator tolerances.
- The AO overlap is nonsingular, the native effective potentials agree with direct molecular-integral evaluation of `J[P_alpha + P_beta] - K[P_sigma]`, and the stored total energy agrees with its AO-state reconstruction.
- The minimum occupied-virtual gap in each spin channel exceeds `orbital_gap_tolerance`, and the complete coupled alpha/beta occupied-virtual operator passes the configured dimension, symmetry, stability, and condition-number gates.
- The projector metadata, descriptor feature count, scalar correction output, double-precision model state, deterministic model sensitivity, and ordered-spectrum differentiability satisfy the shared force-capability contract.
- The runtime PySCF version belongs to the `2.14` series characterized by the compatibility adapter.

The UHF method binding and each response fingerprint the accepted molecular, orbital, occupation, AO-density, total-energy, reference-type, and convergence state, and method calls revalidate that the bound reference, molecule, descriptor, projector, and science-state fingerprint remain unchanged; response controls are recorded and audited separately in `UHFResponseDiagnostics`.

## 4. Spin and descriptor conventions

For `sigma` in `{alpha, beta}`, the accepted occupations are one and `P_sigma = C_sigma,o C_sigma,o.T`; the canonical descriptor density remains `P = P_alpha + P_beta`.

Each projected block is `D_s = O_s.T P O_s`, so `q` is one spin-summed ordered-spectrum descriptor rather than two independently diagonalized spin descriptors.

The correction AO objective is common to both spin channels: `W = partial e_corr / partial P = sum_I,k (partial e_corr / partial q[I,k]) dq_dP[I,k]` evaluated at the total density.

`dq_dR_explicit_spin[sigma]` is an additive decomposition of fixed-density AO/projector motion: its eigenvalue derivative is evaluated from the total-density projected blocks, while its projected-density motion is evaluated with `P_sigma`; therefore `dq_dR_explicit = sum_sigma dq_dR_explicit_spin[sigma]`.

The response decomposition is `dq_dR_response_spin[sigma] = (dq_dP) : P_sigma^R`, `dq_dR_response = sum_sigma dq_dR_response_spin[sigma]`, and `dq_dR_relaxed_spin[sigma] = dq_dR_explicit_spin[sigma] + dq_dR_response_spin[sigma]`.

The complete model-independent descriptor Jacobian is `dq_dR_relaxed = sum_sigma dq_dR_relaxed_spin[sigma] = dq_dR_explicit + dq_dR_response`.

## 5. Coupled UHF response equations

Let `i_sigma` and `a_sigma` index occupied and virtual orbitals in spin channel `sigma`, and let `X_sigma[a_sigma,i_sigma]` be a trial occupied-virtual response amplitude.

The trial AO density is `delta P_sigma[X_sigma] = C_sigma,v X_sigma C_sigma,o.T + C_sigma,o X_sigma.T C_sigma,v.T`.

The coupled induced potentials are `delta V_alpha = J[delta P_alpha + delta P_beta] - K[delta P_alpha]` and `delta V_beta = J[delta P_alpha + delta P_beta] - K[delta P_beta]`.

The physical unshifted coupled operator is `(A X)_sigma = (epsilon_sigma,v - epsilon_sigma,o) X_sigma + C_sigma,v.T delta V_sigma[X_alpha, X_beta] C_sigma,o`, with combined dimension `nvir_alpha*nocc_alpha + nvir_beta*nocc_beta`.

For coordinate `R`, the occupied-space metric response satisfies `U_sigma,oo^R + (U_sigma,oo^R).T = -S_sigma,oo^R`, and its AO-density contribution satisfies `P_sigma,metric^R = -P_sigma S^R P_sigma`.

The coupled occupied-virtual amplitudes solve `A X^R = -B^R`, where `B_sigma^R = C_sigma,v.T (H_sigma^R + delta V_sigma[P_alpha,metric^R, P_beta,metric^R]) C_sigma,o - S_sigma,vo^R epsilon_sigma,o`.

The occupied-virtual density is `P_sigma,occupied_virtual^R = C_sigma,v X_sigma^R C_sigma,o.T + C_sigma,o (X_sigma^R).T C_sigma,v.T`, and the complete numerical AO derivative is `P_sigma^R = P_sigma,metric^R + P_sigma,occupied_virtual^R`.

The independent physical residual is `r_sigma^R = (A X^R)_sigma + B_sigma^R`; its alpha and beta maxima and joint RMS are recomputed from the returned response, fresh direct induced potentials, derivative Hamiltonians, overlap derivatives, orbital energies, and metric contribution.

The total first-order density is `P^R = P_alpha^R + P_beta^R`, and neither a single spin response nor the occupied-virtual term alone has the semantics of `P^R`.

## 6. Runtime API and tensor contract

The following classes and functions are exported from `deepks.deephf`: `UHFDeePHF`, `UHFDeePHFGradients`, `UHFResponse`, `UHFResponseDiagnostics`, `UHFResponseAdapter`, `UHFResponseError`, and `validate_uhf_reference`.

| API | Result |
|---|---|
| `UHFDeePHF(reference, model, projector_basis=..., response_options=...)` | A perturbative energy method bound to one strict native UHF state. |
| `UHFDeePHF.kernel()` | Scalar `e_tot = e_base + e_corr`, while retaining `e_base`, `e_corr`, and `e_tot`. |
| `UHFDeePHF.spin_ao_density()` | Native `(P_alpha, P_beta)` as shape `(2, n_ao, n_ao)`. |
| `UHFDeePHF.ao_density()` | Canonical spin-summed `P` as shape `(n_ao, n_ao)`. |
| `UHFDeePHF.response(**options)` | An audited immutable `UHFResponse` for every nuclear Cartesian coordinate. |
| `UHFDeePHF.first_order_spin_density(response=None, **options)` | Stacked alpha/beta complete AO density derivatives. |
| `UHFDeePHF.first_order_density(response=None, **options)` | Complete spin-summed AO density derivative. |
| `UHFDeePHF.dq_dR_explicit_spin()` | Additive alpha/beta fixed-density descriptor-motion components. |
| `UHFDeePHF.dq_dR_response_spin(response=None, **options)` | Additive alpha/beta density-response descriptor components. |
| `UHFDeePHF.dq_dR_relaxed_spin(response=None, **options)` | Additive alpha/beta complete descriptor-derivative components. |
| `UHFDeePHF.dq_dR_explicit()`, `dq_dR_response(...)`, `dq_dR_relaxed(...)` | Canonical spin-summed descriptor derivatives. |
| `UHFDeePHF.nuc_grad_method(backend="direct", **options)` | A strict `UHFDeePHFGradients` direct driver. |
| `UHFDeePHF.gradient(...)`, `UHFDeePHF.forces(...)` | Complete analytic gradient and its negative. |
| `UHFDeePHFGradients.kernel(atmlst=None)` | Complete gradient for all raw atoms or selected raw-atom indices. |
| `UHFDeePHFGradients.run(atmlst=None)`, `forces(atmlst=None)` | Populated driver or negative gradient under the same selection contract. |
| `UHFResponseAdapter.audit_response_equations(response)` | Full independent reconstruction and equation audit of a supplied response. |

| Value | Runtime axes | Unit |
|---|---|---|
| `spin_ao_density` | `(2, n_ao, n_ao)` | Native dimensionless AO-density convention. |
| Alpha or beta MO-response fields | `(n_raw_atom, 3, n_mo, n_occ_sigma)` | `Bohr^-1`. |
| Alpha or beta coefficient-response fields | `(n_raw_atom, 3, n_ao, n_occ_sigma)` | `Bohr^-1`. |
| Alpha, beta, or total density-response fields | `(n_raw_atom, 3, n_ao, n_ao)` | `Bohr^-1` in the numerical AO representation. |
| `first_order_spin_density` | `(2, n_raw_atom, 3, n_ao, n_ao)` | `Bohr^-1`. |
| `overlap_derivative` | `(n_raw_atom, 3, n_ao, n_ao)` | `Bohr^-1`. |
| Alpha or beta `hamiltonian_derivative` | `(n_raw_atom, 3, n_ao, n_ao)` | `Eh/Bohr`. |
| Alpha or beta `orbital_response_residual` | `(n_raw_atom, 3, n_virtual_sigma, n_occ_sigma)` | `Eh/Bohr`. |
| Spin-resolved descriptor derivatives | `(2, n_raw_atom, 3, n_descriptor_atom, n_feature)` | `Bohr^-1`. |
| Spin-summed descriptor derivatives | `(n_raw_atom, 3, n_descriptor_atom, n_feature)` | `Bohr^-1`. |
| Spin-resolved correction-gradient partitions | `(2, n_raw_atom, 3)` | `Eh/Bohr`. |
| Native, spin-summed correction, and total gradients | `(n_raw_atom, 3)` before optional atom selection | `Eh/Bohr`. |

Every strict response array is real, finite, immutable, C-contiguous `numpy.float64` and is covered by an integrity fingerprint.

## 7. Gradient assembly and retained partitions

For each spin channel, `g_explicit,sigma^R = sum_I,k (partial e_corr / partial q[I,k]) dq_dR_explicit_spin[sigma,R,I,k]` and `g_response,sigma^R = W : P_sigma^R`.

The response contraction is retained as `g_response,sigma = g_metric,sigma + g_occupied_virtual,sigma`, where each term contracts the common `W` with the corresponding spin AO-density response.

The correction gradient satisfies `g_corr = sum_sigma (g_explicit,sigma + g_metric,sigma + g_occupied_virtual,sigma)`, and the complete energy gradient is `g_tot = g_UHF_native + g_corr`.

`UHFDeePHFGradients` publishes `dq_dR_explicit_spin`, `dq_dR_response_spin`, `dq_dR_relaxed_spin`, their spin sums, `correction_gradient_explicit_spin`, `correction_gradient_metric_spin`, `correction_gradient_occupied_virtual_spin`, `correction_gradient_response_spin`, `correction_gradient_spin`, every corresponding spin sum, `reference_gradient`, `de_full`, and the selected `de`.

A zero correction and a nonzero constant correction have zero model sensitivity and therefore reduce the analytic gradient to the native UHF gradient while retaining the corresponding total-energy constant.

## 8. Solver controls, diagnostics, and invariants

| Option | Default | Acceptance meaning |
|---|---:|---|
| `cphf_tolerance` | `1e-11` | Internal tolerance passed to each PySCF UC-PHF solve. |
| `residual_tolerance` | `1e-9` | Maximum accepted independently reconstructed coupled alpha/beta orbital residual. |
| `invariant_tolerance` | `1e-9` | Maximum accepted metric, idempotency, particle-number, reconstruction, and translation residual. |
| `orbital_gap_tolerance` | `1e-7` | Strict lower bound for both spin-channel occupied-virtual gaps in `Eh`. |
| `max_cycle` | `100` | Maximum iterations passed to each PySCF UC-PHF solve. |
| `max_refinement_cycles` | `3` | Maximum residual-correction solves after the initial coupled response. |
| `level_shift` | `0.0` | Solver level shift; the operator audit always examines the unshifted physical operator. |
| `operator_stability_tolerance` | `1e-6` | Strict lower bound for the minimum eigenvalue of the coupled physical operator in `Eh`. |
| `operator_condition_tolerance` | `1e8` | Maximum accepted spectral condition number. |
| `operator_symmetry_tolerance` | `1e-10` | Maximum accepted absolute matrix symmetry residual. |
| `operator_dimension_limit` | `512` | Maximum combined alpha/beta occupied-virtual dimension admitted to the explicit dense audit. |

After the first `pyscf.scf.ucphf.solve` call, the adapter independently reconstructs the physical coupled residual; an excessive residual triggers normalized coupled correction solves and fresh residual evaluations up to `max_refinement_cycles`.

`UHFResponseDiagnostics` records both spin gaps; combined, alpha, beta, and RMS response residuals; the entire refinement history; alpha, beta, and combined response dimensions; operator spectral bounds, condition number, and symmetry residual; alpha and beta occupied-space metric residuals; alpha and beta first-order idempotency and particle-number residuals; density-partition reconstruction; alpha, beta, and total translation residuals; every active control; and the exact PySCF version.

For each spin channel, the independent invariants are `P_sigma^R S P_sigma + P_sigma S^R P_sigma + P_sigma S P_sigma^R - P_sigma^R = 0` and `Tr(P_sigma^R S) + Tr(P_sigma S^R) = 0`.

The adapter also checks alpha and beta metric identities, every MO/coefficient/density partition reconstruction, total-density spin reconstruction, and the translational sums of alpha, beta, and total first-order densities.

## 9. PySCF 2.14 compatibility boundary

All UHF-specific access to PySCF private or semi-private state is isolated in `deepks.deephf.pyscf_uhf`.

That module is the unique owner of `pyscf.hessian.uhf.Hessian.make_h1`, `pyscf.scf.ucphf.solve`, direct molecular J/K construction through `pyscf.scf.hf.get_jk`, exact UHF and Mole state inspection, and the private molecular arrays used by science-state fingerprints.

The adapter converts PySCF-specific values into immutable `UHFResponse` and frozen `UHFResponseDiagnostics` records before the method or gradient layer consumes them.

Architecture tests bind each PySCF facility to its exact RHF or UHF adapter owner, prohibit private PySCF state in non-adapter method modules, enforce unique ownership of the exported UHF direct and scalar-adjoint symbols, and preserve the separation of UHF code from RHF force-data and RKS scalar-adjoint modules.

## 10. Failure behavior and state safety

| Boundary | Explicit behavior |
|---|---|
| Reference type, convergence, molecular physics, decorations, hooks, orbital state, occupations, canonical state, native J/K, energy reconstruction, PySCF series, spin gap, operator dimension, stability, or conditioning | `DeePHFCapabilityError` is raised before a response or force result is returned. |
| Projector metadata, model dtype/state/output/sensitivity, model determinism, or descriptor differentiability | The shared capability or descriptor error propagates before response contraction. |
| PySCF derivative construction, UC-PHF solve or refinement, nonfinite response, operator asymmetry, excessive independent residual, invariant failure, or reconstruction failure | `UHFResponseError` is raised without a density, descriptor, or gradient fallback. |
| Supplied response with a foreign identity, stale state, invalid exact type, changed integrity digest, mutable/non-double/nonfinite array, inconsistent partition, forged diagnostic, changed controls, or failed equation audit | `UHFResponseError` is raised before the response is consumed. |
| Invalid response-control scalar, tolerance, or cycle limit | `ValueError` is raised while constructing the adapter. |
| A supplied response combined with response-option keywords | `ValueError` is raised before either input can be ignored. |
| A backend name other than `direct` or `zvector`, an option from the other backend namespace, or UHF gradient-scanner construction | A capability, response, or adjoint error is raised at the requested boundary. |
| Invalid atom selection or corrupted method-driver binding | The selection or binding error is raised before response publication. |

`UHFDeePHFGradients.kernel` clears every public result before evaluation and clears them again on any failure, so an unsuccessful call cannot expose a preceding response, partition, or gradient.

The UHF direct path never substitutes `dq_dR_explicit`, an RHF response, or a scalar adjoint for a missing or failed coupled coordinate-wise UHF response.

## 11. P3A and P3B boundaries

The P3A persistence API is explicitly RHF: `generate_rhf_force_frame` and `write_rhf_force_dataset` validate exact native RHF references and store the RHF direct oracle's model-independent `dq_dR_relaxed` contract.

The P3B scalar-adjoint classes, Z-vector gradient driver, and fresh-reference scanner are explicitly RHF and accept exact RHF method state under their existing contracts.

`UHFDeePHF`, `UHFResponse`, `UHFDeePHFGradients`, `UHFAdjoint`, and `UHFDeePHFZVectorGradients` are distinct UHF inference types; they do not enter the RHF force-data producer or RHF adjoint/scanner object graph. The UHF scalar-adjoint contract is documented in [P4D UHF DeePHF Z-vector inference](./p4d_uhf_zvector_inference.md).

Architecture and runtime tests enforce these type and dependency boundaries in both directions.

## 12. Deterministic acceptance and commands

The numerical oracle uses a symmetry-disabled, spherical, distorted neutral `NH2/STO-3G` doublet with spin `1`, fixed coordinates in Bohr, projector shells `[[0, [0.8, 1.0]], [1, [0.3, 1.0]]]`, and a deterministic nontrivial double-precision nonlinear correction model.

Every central displacement independently converges a fresh native UHF reference with `conv_tol=1e-13`, `conv_tol_grad=1e-10`, `conv_tol_cpscf=1e-12`, and `max_cycle=100`; the sequence preserves per-spin occupations and AO labels, keeps the reference-to-displaced occupied-subspace minimum singular value above `0.99` in each spin channel, and retains finite positive spin gaps and a differentiable descriptor.

Central differences at `(1e-3, 3e-4, 1e-4) Bohr` independently validate alpha, beta, and total AO density derivatives, the complete relaxed descriptor derivative, and the complete `e_base + e_corr` gradient.

Independent AO-integral oracles validate every block and spectral diagnostic of the coupled alpha/beta occupied-virtual operator, the metric density in each spin channel, the overlap derivative, and the complete physical coupled residual.

The acceptance suite also verifies all spin and motion identities, omission detectability, translation, first-order nonorthogonal invariants, immutable supplied-response audits, residual refinement and fault injection, strict failures, result clearing, native-reference immutability, atom selection, force sign, zero and constant corrections, and P3A/P3B isolation.

Run the accepted verification sequence from the repository root:

```bash
uv sync --locked --python 3.11
uv run pytest tests/uhf_analytic_forces
uv run pytest tests/zvector_inference
uv run pytest tests/analytic_forces
uv run pytest tests/force_training
uv run pytest tests/baseline
uv run pytest
uv build
git diff --check
```
