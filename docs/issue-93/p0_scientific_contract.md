# P0 Scientific Contract

## 1. Purpose

This document defines the scientific meaning, supported domain, package boundaries, canonical vocabulary, tensor conventions, and failure behavior for the Issue #93 implementation.

An analytic force claim means the exact first nuclear derivative of the stated approximate method within this contract and within the reported numerical tolerances.

Every public value must have one method, derivative meaning, sign, unit, and axis order; a value with weaker semantics must use a different name.

## 2. Method definitions

### 2.1 Self-consistent DeePKS

`deepks.deepks` implements self-consistent DeePKS, in which the correction potential participates in the SCF equations and the converged density is stationary for the total corrected functional.

For an accepted converged, differentiable, penalty-free DeePKS state, the total first derivative is evaluated by the native variational gradient plus the explicit correction derivative.

Penalty behavior belongs only to `deepks.deepks`; a penalty-bearing state has no strict analytic-force capability unless the reported energy contains the matching penalty functional and its complete derivative.

### 2.2 Perturbative DeePHF

`deepks.deephf` implements perturbative DeePHF by composition around an already converged native reference and never injects the correction into the reference Fock matrix, orbitals, occupations, convergence state, or scanner lifecycle.

The perturbative energy is `e_tot(R) = e_base(P(R), R) + e_corr(q(P(R), O(R)))`, where `P(R)` is the independently stationary reference density.

The DeePHF correction is not stationary with respect to the reference orbitals, so its exact first derivative requires both explicit descriptor motion and reference-density response.

### 2.3 Method-neutral data

`deepks.data` defines method-neutral field selection, molecular input, array serialization, and statistics facilities that consume stable method outputs and do not import either method package's internal modules.

A stored quantity retains the same mathematical meaning across calculation, serialization, reading, training, validation, and saved-data testing.

## 3. Package and dependency contract

The package roles are:

- `deepks.descriptor` owns projector construction, projection integrals, projected-density evaluation, spectral descriptor evaluation, local explicit nuclear derivatives, AO-density derivatives, atom-index mappings, and descriptor differentiability diagnostics.

- `deepks.deepks` owns self-consistent correction potentials, corrected SCF classes, variational gradients, penalty behavior, stable field-producing methods, and the DeePKS runner.

- `deepks.deephf` owns perturbative composition, reference capability validation, response backends, perturbative gradients, and DeePHF-specific result production.

- `deepks.model` owns correction-model definitions, shared model evaluation from descriptor values, training-data readers, training, and saved-data evaluation.

- `deepks.data` owns neutral field definitions, molecular input, array writers, and statistics without knowledge of concrete method classes.

- CLI, task, iteration, and workflow layers may select a method explicitly and may depend on public method and data interfaces.

The permitted dependency direction is:

```text
utils and contracts
        |
        +----> descriptor
        |
        +----> model
                  \
descriptor --------+----> deepks
                  \
                   +----> deephf

public method results ----> data ----> training, CLI, tasks, and workflows
```

The following dependency rules are mandatory:

- `deepks.descriptor` imports neither `deepks.deepks` nor `deepks.deephf` and contains no SCF correction, penalty, training, field, CLI, or workflow behavior.

- `deepks.deepks` and `deepks.deephf` never import one another.

- Shared descriptor mathematics is implemented once in `deepks.descriptor` and is consumed by both methods.

- Shared model evaluation is implemented once in `deepks.model` and does not depend on either method package.

- `deepks.data` does not import method implementation modules; method objects provide every requested value through stable public methods.

- No package provides an alias, import shim, field alias, or configuration fallback for the mixed `deepks.scf` layout.

## 4. Canonical mathematical vocabulary

The canonical symbols and identifiers are:

| Symbol | Canonical identifier | Meaning |
|---|---|---|
| `P` | `P` | Spin-summed AO one-particle density matrix in the native PySCF AO convention. |
| `P_alpha`, `P_beta` | `spin_ao_density` components | Alpha and beta AO one-particle densities whose sum is the canonical `P` for UHF. |
| `O` | `O` | AO-to-projector overlap tensor, partitioned by projector shell. |
| `D` | `D` | Projected-density block `D_s = O_s^T P O_s`. |
| `q` | `q` | Descriptor obtained by concatenating the ascending eigenvalues of every `D_s` block. |
| `W` | `correction_ao_potential` | Complete correction objective `partial e_corr / partial P`, formed by contracting model sensitivity with `dq_dP`. |
| `A` | response operator | Physical occupied-virtual reference-response operator in `A X^R = -B^R`. |
| `b` | `objective_orbital_gradient` | Bilateral occupied-virtual derivative of the scalar correction objective. |
| `z` | `zvector` | Scalar adjoint satisfying the literal transpose equation `A.T z = b`. |
| `dq_dR_explicit` | `dq_dR_explicit` | Nuclear derivative of `q` at fixed numerical AO density matrix `P`, including AO-center and projector-center motion. |
| `dq_dR_response` | `dq_dR_response` | Descriptor derivative caused by the complete first-order reference density, `(partial q / partial P) : P^R`. |
| `dq_dR_relaxed` | `dq_dR_relaxed` | Complete derivative `dq_dR_explicit + dq_dR_response`. |
| `dq_dR_explicit_spin` | `dq_dR_explicit_spin` | Additive alpha and beta components of fixed-density UHF descriptor motion, evaluated with total-density descriptor eigenvectors. |
| `dq_dR_response_spin` | `dq_dR_response_spin` | Additive UHF descriptor-response components `(partial q / partial P) : P_sigma^R`. |
| `dq_dR_relaxed_spin` | `dq_dR_relaxed_spin` | Additive UHF components `dq_dR_explicit_spin + dq_dR_response_spin` whose spin sum is `dq_dR_relaxed`. |
| `e_base` | `e_base` | Native reference or base-functional energy evaluated at the method state. |
| `e_corr` | `e_corr` | Learned correction energy evaluated from `q`. |
| `e_tot` | `e_tot` | Total reported energy `e_base + e_corr`. |
| Correction-energy target | `e_corr_target` | Reference energy label minus `e_base`; this is supervision data and is distinct from the model-evaluated `e_corr`. |
| DeePKS reference partition | `f_reference_variational` | Negative native variational gradient evaluated at the converged DeePKS density; this is a computational partition, not the complete derivative of `e_base` along the DeePKS state path. |
| DeePKS explicit correction partition | `f_corr_explicit` | Negative fixed-density explicit correction gradient obtained by contracting `dq_dR_explicit` with `partial e_corr / partial q`; this is the model-evaluated value paired with `dq_dR_explicit`. |
| `f_tot` | `f_tot` | Complete total force `-d(e_tot)/dR` for the named method. |
| Explicit correction-force target | `f_corr_explicit_target` | Reference force label minus `f_reference_variational`; this is supervision data and is distinct from the model-evaluated `f_corr_explicit`. |

`P`, `O`, `D`, and `q` are the canonical tensor names in mathematical and response-sensitive code; explanatory APIs may add an unambiguous noun but must not introduce a second stored-field vocabulary.

The terms `gradient` and `force` are not interchangeable: a gradient is `dE/dR`, while a force is `-dE/dR`.

A DeePKS backend exposes `f_tot`, `f_reference_variational`, and `f_corr_explicit`; it does not label either computational partition as the complete derivative of an individual energy component along the converged state path.

The fixed-density quantity `dq_dR_explicit` must never be read, written, passed, or interpreted as `dq_dR_relaxed`.

## 5. Descriptor definition and atom mapping

Let `A` index every raw molecular atom center, `x` index Cartesian components in the order `(x, y, z)`, `I` index descriptor atoms, `mu` and `nu` index molecular AOs, `s` index projector shells, `m` and `n` index functions within a projector shell, and `k` index the concatenated descriptor components.

For RHF, `P[mu,nu] = sum_i C[mu,i] mo_occ[i] C[nu,i]` with accepted occupations equal to zero or two.

For UHF, `P_sigma[mu,nu] = sum_i C_sigma[mu,i] mo_occ_sigma[i] C_sigma[nu,i]` with accepted occupations equal to zero or one, and the descriptor uses `P = P_alpha + P_beta`.

For each projector shell, `O_s[mu,I,m] = <chi_mu | alpha_sIm>`, `D_s[I,m,n] = sum_mu,nu O_s[mu,I,m] P[mu,nu] O_s[nu,I,n]`, and `q` concatenates `eigvalsh(D_s[I])` in configured shell order and ascending eigenvalue order within each block.

The raw-atom-to-descriptor-atom mapping is explicit, stable for a geometry, and stored with results that contain a descriptor atom axis.

## 6. Ghost-center contract

The only canonical ghost predicate is `mol.atom_charge(A) == 0`; element-name prefixes and aliases are not scientific identity tests.

The raw atom axis contains every molecular center, including ghosts, while the descriptor atom axis contains exactly the raw atoms for which `mol.atom_charge(A) != 0`.

AO-center motion is evaluated for every raw atom with an AO slice, including a ghost carrying basis functions.

Projector-center motion is evaluated only for descriptor atoms, using the explicit raw-to-descriptor index mapping.

Elemental correction constants exclude ghosts through the same nuclear-charge predicate.

A ghost coordinate therefore retains a row in every nuclear derivative and force tensor even though the ghost has no descriptor-atom entry of its own.

The shared descriptor accepts a ghost-containing molecular input when its AO slices and descriptor mapping satisfy these rules and the explicit derivative checks pass; the strict RHF, UHF, and RKS DeePHF reference validators admit real-atom molecular references.

## 7. Tensor axes, signs, dtypes, and units

Runtime tensors omit a frame axis; serialized arrays prepend `n_frame` without changing any remaining axis order.

| Quantity | Runtime axes | Serialized axes | Unit |
|---|---|---|---|
| `P` | `(n_ao, n_ao)` | `(n_frame, n_ao, n_ao)` | Native dimensionless AO density convention. |
| `P_alpha`, `P_beta`; stacked `spin_ao_density` | `(n_ao, n_ao)` each; `(2, n_ao, n_ao)` stacked | Diagnostic runtime only. | Native dimensionless AO density convention. |
| `O_s` | `(n_ao, n_descriptor_atom, n_projector_s)` | Not a required public field. | Dimensionless overlap. |
| `D_s` | `(n_descriptor_atom, n_projector_s, n_projector_s)` | Shell-aware storage only. | Dimensionless. |
| `q` | `(n_descriptor_atom, n_projector)` | `(n_frame, n_descriptor_atom, n_projector)` | Dimensionless. |
| `dq_dR_explicit` | `(n_raw_atom, 3, n_descriptor_atom, n_projector)` | `(n_frame, n_raw_atom, 3, n_descriptor_atom, n_projector)` | Runtime `Bohr^-1`; serialized inverse declared molecular length unit. |
| `dq_dR_response` | `(n_raw_atom, 3, n_descriptor_atom, n_projector)` | Diagnostic storage only. | Runtime `Bohr^-1`; serialized inverse declared molecular length unit. |
| `dq_dR_relaxed` | `(n_raw_atom, 3, n_descriptor_atom, n_projector)` | `(n_frame, n_raw_atom, 3, n_descriptor_atom, n_projector)` | Runtime `Bohr^-1`; serialized inverse declared molecular length unit. |
| UHF `dq_dR_explicit_spin`, `dq_dR_response_spin`, `dq_dR_relaxed_spin` | `(2, n_raw_atom, 3, n_descriptor_atom, n_projector)` | Diagnostic runtime only. | `Bohr^-1`. |
| UHF `first_order_spin_density` | `(2, n_raw_atom, 3, n_ao, n_ao)` | Diagnostic runtime only. | `Bohr^-1` in the numerical AO representation. |
| RKS `first_order_density` | `(n_raw_atom, 3, n_ao, n_ao)` | Diagnostic runtime only. | `Bohr^-1` in the numerical AO representation. |
| RKS overlap derivative | `(n_raw_atom, 3, n_ao, n_ao)` | Diagnostic runtime only. | `Bohr^-1`. |
| RKS fixed-grid, XC AO-motion, grid-coordinate, and grid-weight Hamiltonian derivatives | `(n_raw_atom, 3, n_ao, n_ao)` | Diagnostic runtime only. | `Eh/Bohr`. |
| `W` | `(n_ao, n_ao)` | Direct- or scalar-inference runtime result. | `Eh`. |
| `b`, `z` | `(n_virtual, n_occupied)` for closed-shell RHF or RKS; separate alpha and beta matrices for UHF | Scalar-adjoint runtime result. | `b` in `Eh`; `z` dimensionless. |
| `e_base`, `e_corr`, `e_tot`, `e_corr_target` | scalar | `(n_frame, 1)` | `Eh`. |
| `f_reference_variational`, `f_corr_explicit`, `f_corr_explicit_target`, `f_tot` | `(n_raw_atom, 3)` | `(n_frame, n_raw_atom, 3)` | Runtime `Eh/Bohr`; serialized `Eh` per declared molecular length unit. |
| UHF spin-resolved correction-gradient partitions | `(2, n_raw_atom, 3)` | Diagnostic runtime only. | `Eh/Bohr`. |
| RKS native fixed-grid, grid-coordinate, grid-weight, correction, and total gradient partitions | `(n_raw_atom, 3)` | Diagnostic runtime only. | `Eh/Bohr`. |

Scientific calculations use real double precision; a different dtype is outside strict capability unless independently characterized and declared.

The shared derivative API returns atomic-unit derivatives, while calculation fields convert once to the molecule's declared coordinate unit so stored forces, descriptor Jacobians, and supplied force labels use the same length convention.

For DeePHF, `d(e_corr)/dR[A,x] = sum_I,k (partial e_corr / partial q[I,k]) dq_dR_relaxed[A,x,I,k]`, and the correction-force contribution is the negative of this contraction.

The response component uses a complete basis-aware `P^R`, including occupied-virtual response and occupied-occupied metric response; an occupied-virtual amplitude tensor alone is not `P^R`.

### 7.1 RHF direct and scalar-adjoint identity

For the accepted spin-summed closed-shell convention, occupied orbitals have `n_i = 2`, `W[mu,nu] = sum_I,k (partial e_corr / partial q[I,k]) dq_dP[I,k,mu,nu]`, and `b[a,i] = n_i (W_mo[a,i] + W_mo[i,a]) = 4 W_mo[a,i]` for symmetric `W`.

For a virtual-occupied amplitude `X`, `delta P(X) = C_v X (C_o diag(n_i)).T + C_o diag(n_i) X.T C_v.T`, `G[delta P] = J[delta P] - 0.5 K[delta P]`, and `(A X)[a,i] = (epsilon_a - epsilon_i) X[a,i] + (C_v.T G[delta P(X)] C_o)[a,i]`.

The direct equation is `A X^R = -B^R`, with `B^R = B_bare^R + B_metric^R`, `B_bare^R[a,i] = h^R[a,i] - epsilon_i S^R[a,i]`, and `B_metric^R[a,i] = (C_v.T G[P_metric^R] C_o)[a,i]`.

The scalar adjoint solves `A.T z = b` once. With `D_z = C_v z (C_o diag(n_i)).T + C_o diag(n_i) z.T C_v.T`, `V_z = G[D_z]`, and `Wbar_oo = 0.5 (W_oo + W_oo.T)`, the exact response partitions are `g_metric^R = -2 S_oo^R : Wbar_oo`, `g_adjoint_nuclear^R = -z : B_bare^R`, `g_adjoint_metric^R = 0.5 S_oo^R : V_z,oo`, `g_occupied_virtual^R = g_adjoint_nuclear^R + g_adjoint_metric^R`, and `g_response^R = g_metric^R + g_occupied_virtual^R`.

The complete correction gradient is `g_corr^R = sum_I,k (partial e_corr / partial q[I,k]) dq_dR_explicit[R,I,k] + g_response^R`, and the complete method gradient is `g_tot^R = g_reference^R + g_corr^R`.

### 7.2 UHF direct-response identity

For `sigma` in `{alpha, beta}`, `P_sigma = C_sigma,o C_sigma,o.T`, `P = P_alpha + P_beta`, and an occupied-virtual trial amplitude gives `delta P_sigma(X_sigma) = C_sigma,v X_sigma C_sigma,o.T + C_sigma,o X_sigma.T C_sigma,v.T`.

The coupled induced potentials are `delta V_alpha = J[delta P_alpha + delta P_beta] - K[delta P_alpha]` and `delta V_beta = J[delta P_alpha + delta P_beta] - K[delta P_beta]`, so `(A X)_sigma = (epsilon_sigma,v - epsilon_sigma,o) X_sigma + C_sigma,v.T delta V_sigma C_sigma,o` acts on the combined alpha/beta occupied-virtual space.

For coordinate `R`, `U_sigma,oo^R + (U_sigma,oo^R).T = -S_sigma,oo^R`, `P_sigma,metric^R = -P_sigma S^R P_sigma`, and the coupled direct equation is `A X^R = -B^R` with `B_sigma^R = C_sigma,v.T (H_sigma^R + delta V_sigma[P_alpha,metric^R, P_beta,metric^R]) C_sigma,o - S_sigma,vo^R epsilon_sigma,o`.

The complete spin density derivative is `P_sigma^R = P_sigma,metric^R + C_sigma,v X_sigma^R C_sigma,o.T + C_sigma,o (X_sigma^R).T C_sigma,v.T`, and the canonical density derivative is `P^R = P_alpha^R + P_beta^R`.

The UHF descriptor identities are `dq_dR_response_spin[sigma] = (partial q / partial P) : P_sigma^R`, `dq_dR_relaxed_spin = dq_dR_explicit_spin + dq_dR_response_spin`, and `dq_dR_relaxed = sum_sigma dq_dR_relaxed_spin[sigma]`.

The common total-density objective `W = partial e_corr / partial P` contracts each spin response, and `g_tot^R = g_UHF^R + sum_sigma W : P_sigma^R + sum_I,k (partial e_corr / partial q[I,k]) dq_dR_explicit^R[I,k]`.

### 7.3 UHF scalar-adjoint identity

The spin-summed descriptor gives the same symmetric AO objective `W` in both spin channels, with `b_sigma[a,i] = W_sigma[a,i] + W_sigma[i,a] = 2 W_sigma[a,i]` under the accepted unit occupations.

The scalar adjoint solves the complete coupled equation `A.T [z_alpha, z_beta] = [b_alpha, b_beta]` once. Its alpha and beta adjoint densities are `D_z,sigma = C_sigma,v z_sigma C_sigma,o.T + C_sigma,o z_sigma.T C_sigma,v.T`, with induced potentials `V_z,alpha = J[D_z,alpha + D_z,beta] - K[D_z,alpha]` and `V_z,beta = J[D_z,alpha + D_z,beta] - K[D_z,beta]`.

For each spin, `g_metric,sigma^R = -S_sigma,oo^R : Wbar_sigma,oo`, `g_adjoint_nuclear,sigma^R = -z_sigma : (H_sigma,vo^R - S_sigma,vo^R epsilon_sigma,o)`, `g_adjoint_metric,sigma^R = 0.5 S_sigma,oo^R : V_z,sigma,oo`, and `g_occupied_virtual,sigma^R = g_adjoint_nuclear,sigma^R + g_adjoint_metric,sigma^R`.

The exact total identities are `g_response = sum_sigma (g_metric,sigma + g_occupied_virtual,sigma)`, `g_corr = g_explicit + g_response`, and `g_tot = g_UHF_native + g_corr`. The spin-resolved adjoint occupied-virtual values are coupled equation-channel contractions, while their sum equals the complete P4A direct occupied-virtual contraction.

### 7.4 RKS direct-response identity

For the accepted closed-shell pure-LDA convention, `P = C_o diag(n_i) C_o.T` with `n_i = 2`, and `delta P(X) = C_v X (C_o diag(n_i)).T + C_o diag(n_i) X.T C_v.T`.

The finite-grid induced potential is `G_RKS[delta P] = J[delta P] + K_xc[delta P]`, where `K_xc` is the dense contraction of the LibXC LDA kernel with the induced density on the exact accepted atom-centered grid.

The physical operator is `(A X)[a,i] = (epsilon_a - epsilon_i) X[a,i] + (C_v.T G_RKS[delta P(X)] C_o)[a,i]`; the implemented dense audit contains both the Coulomb and `f_xc` blocks.

For coordinate `R`, `P_metric^R = -0.5 P S^R P`, and the occupied-virtual equation uses the complete nuclear Hamiltonian derivative `H^R = H_fixed_grid^R + H_xc_grid_coordinate^R + H_xc_grid_weight^R`, the overlap term, and `G_RKS[P_metric^R]`.

The complete density derivative is `P^R = P_metric^R + P_occupied_virtual^R`, the descriptor identities are `dq_dR_response = (partial q / partial P) : P^R` and `dq_dR_relaxed = dq_dR_explicit + dq_dR_response`, and `g_tot^R = g_RKS_native^R + W : P^R + sum_I,k (partial e_corr / partial q[I,k]) dq_dR_explicit^R[I,k]`.

The native finite-grid gradient satisfies `g_RKS_native = g_without_grid_response + g_xc_grid_coordinate + g_xc_grid_weight`, and the complete accepted native term uses PySCF grid response.

### 7.5 RKS scalar-adjoint identity

For the accepted RKS objective, `b[a,i] = n_i (W_mo[a,i] + W_mo[i,a])`, and the scalar adjoint solves the same physical finite-grid operator equation `A.T z = b` once.

With `D_z = C_v z (C_o diag(n_i)).T + C_o diag(n_i) z.T C_v.T`, `V_z = G_RKS[D_z]`, and `Wbar_oo = 0.5 (W_oo + W_oo.T)`, the exact metric terms are `g_metric^R = -2 S_oo^R : Wbar_oo` and `g_adjoint_metric^R = 0.5 S_oo^R : V_z,oo`.

The complete finite-grid nuclear contraction is partitioned as `g_adjoint_fixed_grid^R = -z : (H_fixed_grid,vo^R - S_vo^R epsilon_o)`, `g_adjoint_grid_coordinate^R = -z : H_xc_grid_coordinate,vo^R`, and `g_adjoint_grid_weight^R = -z : H_xc_grid_weight,vo^R`.

The exact identities are `g_adjoint_nuclear = g_adjoint_fixed_grid + g_adjoint_grid_coordinate + g_adjoint_grid_weight`, `g_occupied_virtual = g_adjoint_nuclear + g_adjoint_metric`, `g_response = g_metric + g_occupied_virtual`, `g_corr = g_explicit + g_response`, and `g_tot = g_RKS_native + g_corr`.

## 8. Descriptor degeneracy contract

The ordered-eigenvalue descriptor is accepted only where its nuclear-coordinate derivative is well defined for the evaluated path and the correction model is compatible with every accepted repeated subspace.

A single absolute eigenvalue-gap cutoff is not an acceptance rule.

Every spectral block is classified using recorded scale-aware zero, gap, rank, and finite-difference tolerances.

A simple, isolated eigenvalue block is accepted when eigenvalue ordering and analytic derivatives remain stable over the configured central-displacement sequence.

A repeated zero block is accepted only when it is a structural fixed-rank null space: the rank is stable, the null eigenvalues remain zero within tolerance, no positive eigenvalue crosses into the block, and directional derivative checks remain stable over the displacement sequence.

A symmetry-induced splitting, state crossing, changing rank, unstable ordering, or near crossing that prevents a stable ordered-eigenvalue derivative is rejected.

Model compatibility is evaluated after all preprocessing, normalization, embeddings, shell weights, and linear branches; sensitivities within an accepted repeated subspace must be equal within the recorded tolerance whenever subspace invariance is required.

Checkpoint format or model class identity is not evidence of degeneracy compatibility; the evaluated model behavior is the criterion.

A contracted energy derivative that happens to be finite does not authorize storage of an ambiguous individual `dq_dR_relaxed`; force-data generation requires the descriptor Jacobian itself to satisfy this contract.

Differentiability diagnostics, tolerances, block classification, and the displacement sequence are part of the calculation provenance.

## 9. Strict reference capabilities

The current strict DeePHF facilities are:

| Reference family | Current facility | Public entry point | Scientific result |
|---|---|---|---|
| Native RHF | Coordinate-wise direct oracle | `DeePHF.response()` and `RHFDeePHFGradients` | Complete RHF `P^R`, `dq_dR_relaxed`, and exact `e_base + e_corr` gradient. |
| Native RHF | Scalar-adjoint inference | `DeePHF.adjoint()` and `RHFDeePHFZVectorGradients` | One correction-specific transpose solve and exact scalar correction gradient. |
| Native RHF | Fresh-reference scanning | `RHFDeePHFGradientScanner` | Atomic fresh-reference energy and direct or Z-vector gradient publication across accepted geometries. |
| Native RHF | Relaxed-force data and training | `generate_rhf_force_frame`, `write_rhf_force_dataset`, and strict force-data consumers | Persisted model-independent RHF `dq_dR_relaxed` with force provenance. |
| Native UHF | Coordinate-wise direct oracle | `UHFDeePHF.response()` and `UHFDeePHFGradients` | Complete alpha/beta and total `P^R`, additive spin descriptor partitions, and exact `e_base + e_corr` gradient. |
| Native UHF | Scalar-adjoint inference | `UHFDeePHF.adjoint()` and `UHFDeePHFZVectorGradients` | One coupled alpha/beta transpose solve and exact scalar correction gradient. |
| Native closed-shell RKS pure LDA | Coordinate-wise direct oracle | `RKSDeePHF.response()` and `RKSDeePHFGradients` | Complete finite-grid `P^R`, `dq_dR_relaxed`, native grid-response partitions, and exact `e_base + e_corr` gradient. |
| Native closed-shell RKS pure LDA | Scalar-adjoint inference | `RKSDeePHF.adjoint()` and `RKSDeePHFZVectorGradients` | One correction-specific finite-grid transpose solve and exact scalar correction gradient. |

Every strict force facility requires a fixed compatible projector, one real scalar correction, real double-precision finite model state, deterministic and finite complete model sensitivity, accepted ordered-spectrum differentiability, and an unchanged fingerprintable scientific state throughout the transaction; the RKS facility additionally binds the normalized LibXC functional and deterministic finite grid.

### 9.1 Strict RHF reference

The RHF validator accepts an exact converged `pyscf.scf.hf.RHF` attached to an exact molecular `pyscf.gto.mole.Mole`, with spin zero, complete real canonical orbitals, occupations exactly zero or two in the Aufbau ground-state root, both occupied and virtual spaces, and an internally consistent AO density, Fock state, electron count, canonical residual, and total energy.

The accepted molecule is finite, symmetry-disabled, nonperiodic, all-electron, point-nuclear, and real-atom; it uses spherical molecular Gaussian AOs and the full Coulomb interaction, and its reference and molecule have no active decoration or callable instance-hook state.

The RHF direct and scalar-adjoint operators independently pass their configured gap, dimension, symmetry, stability, condition, finite-value, and residual gates under the PySCF 2.14 RHF adapter.

### 9.2 Strict UHF reference

The UHF validator accepts an exact converged `pyscf.scf.uhf.UHF` attached to an exact molecular `pyscf.gto.mole.Mole`, with complete real canonical alpha and beta orbitals, occupations exactly zero or one in each Aufbau ground-state spin root, electron counts matching `mol.nelec` and `mol.spin`, occupied and virtual spaces in both channels, and internally consistent spin AO densities, effective potentials, canonical residuals, and total energy.

The accepted molecule has the same finite molecular, spherical, symmetry-disabled, all-electron, point-nuclear, real-atom, full-Coulomb, undecorated, and hook-free properties as the strict RHF molecule.

The complete combined alpha/beta occupied-virtual operator independently passes its configured spin-gap, dimension, symmetry, stability, and condition gates. The coordinate-wise direct response additionally passes alpha, beta, joint, metric, idempotency, particle-number, reconstruction, and translation audits, while the scalar adjoint passes objective, operator, induced-density, induced-potential, transpose-equation, physical-equation, gradient-reconstruction, provenance, and integrity audits under the PySCF 2.14 UHF adapter.

### 9.3 Strict RKS direct reference

The RKS validator accepts an exact converged `pyscf.dft.rks.RKS` attached to an exact molecular `pyscf.gto.mole.Mole`, with spin zero, an even electron count, complete real canonical orbitals, occupations exactly zero or two in the Aufbau ground-state root, occupied and virtual spaces, and an internally consistent spin-summed AO density, finite-grid effective potential, canonical residual, and total energy.

The accepted molecule is finite, spherical, symmetry-disabled, all-electron, point-nuclear, real-atom, full-Coulomb, undecorated, and hook-free.

The accepted functional uses the characterized native LibXC `7.0.0` backend with `NumInt.cutoff=1e-13`, normalizes to components `((1, 1.0), (7, 1.0))` for pure `LDA_X + LDA_C_VWN`, has the canonical LibXC parameter signature, and has no hybrid, range-separated, NLC, or custom-functional contribution.

The accepted exact native grid is prebuilt and byte-reproducible with `(20, 50)` atom grids for every element, no pruning or density cutoff, native Treutler-Ahlrichs radial and radii-adjustment functions, original Becke partitioning, alignment one, grid cutoff `1e-15`, unsorted grid order, and canonical finite float64 `(131,)` `BRAGG_RADII` fingerprint `d5eeefc53bb8261154cd2317ff60e5e642dd9cde1d1f283647b7956756b74a43`.

The exact `pyscf.grad.rks.grids_response_cc` identity and one contiguous block per cached host-atom range are mandatory; a block contains finite float64 coordinates `(n_host_point, 3)`, weights `(n_host_point,)`, and `w1` `(n_raw_atom, 3, n_host_point)`, coordinates are exact cached-grid values, and approximately `1e-198` PySCF tail-weight differences are admitted only by the `rtol=0`, `atol=1e-180` boundary before all contractions use cached energy-grid weights.

Concatenated `w1` has shape `(n_raw_atom, 3, n_grid)`, satisfies nuclear translation to `1e-10`, agrees with independent strict-grid central finite differences at `h=1e-5 Bohr` with `rtol=1e-7` and `atol=1e-6`, and contributes the qualified response-generator identity and weight-derivative fingerprint to immutable grid provenance.

The RKS occupied-virtual operator independently passes its configured gap, dimension, symmetry, stability, and condition gates, and each coordinate-wise response passes Coulomb plus dense `f_xc`, fixed-grid XC, grid-coordinate, grid-weight, physical residual, metric, idempotency, particle-number, reconstruction, and translation audits under the PySCF 2.14 RKS adapter.

### 9.4 Backend and data boundaries

`deepks.deephf` exposes `direct` and `zvector` as distinct RHF analytic-gradient backends, with `direct` remaining the default for `nuc_grad_method`, `gradient`, and `forces`. The RHF direct backend constructs complete coordinate-wise density and relaxed descriptor responses, while the RHF Z-vector backend evaluates one model-specific scalar correction gradient without constructing either response tensor.

`UHFDeePHF` exposes distinct `direct` and `zvector` backends, with `direct` remaining the default. `UHFDeePHFGradients` constructs the complete coupled alpha/beta coordinate-wise response, while `UHFDeePHFZVectorGradients` performs one model-specific coupled scalar-adjoint solve without constructing the coordinate-wise density or relaxed descriptor response.

`RKSDeePHF` exposes distinct `direct` and `zvector` backends, with `direct` remaining the default. `RKSDeePHFGradients` constructs the complete closed-shell finite-grid coordinate-wise response, while `RKSDeePHFZVectorGradients` performs one model-specific scalar-adjoint solve without constructing the coordinate-wise density or relaxed descriptor response.

One `RKSDeePHF` method retains and independently reaudits at most eight successful responses that it produced, so an earlier retained response remains reusable after a later solve while foreign, changed, or evicted responses fail explicitly.

`DeePHF.response_options`, `UHFDeePHF.response_options`, and `RKSDeePHF.response_options` configure their separate direct adapters, while `DeePHF.adjoint_options`, `UHFDeePHF.adjoint_options`, and `RKSDeePHF.adjoint_options` independently configure their scalar-adjoint backends; each option namespace is validated at its own method and driver boundary.

The reference-neutral `ScalarAdjointProblem` contract supplies `dimension`, `dense_operator`, `apply`, and `apply_transpose`; `solve_scalar_adjoint` performs one literal `A.T z = b` solve and retains literal, independent-transpose, and physical residual diagnostics, while reference-specific operator semantics remain inside the RHF, UHF, and RKS compatibility adapters.

The force-data producer selects the RHF direct backend because persistent `dq_dR_relaxed` is a model-independent coordinate-wise RHF descriptor Jacobian. Scalar `RHFAdjoint`, `UHFAdjoint`, and `RKSAdjoint` results are model-specific inference state with no relaxed-Jacobian semantics, and UHF and RKS runtime objects remain outside the RHF force-data contract.

### 9.5 Strict RHF scanner lifecycle

An RHF DeePHF gradient driver creates a fresh-reference scanner with a fixed `direct` or `zvector` backend, independently copied direct, adjoint, and driver option namespaces, a fixed projector definition, and a continuous occupied-subspace root anchor.

Every scanner call validates atom selection before SCF, constructs a new exact native RHF reference with `dm0=None`, rebuilds the DeePHF method, descriptor, backend response state, and gradient driver, and compares the accepted occupied space with the preceding root through the minimum singular value of the cross-geometry occupied overlap.

The scanner publishes energy, gradient, reference, method, and driver state atomically and advances the root anchor only after the entire call succeeds. A failed call clears its current public result and preserves the preceding accepted root anchor, so stale energy, gradient, response, adjoint, or geometry state is never returned.

## 10. Failure boundaries

Unsupported reference capability, unconverged SCF state, inconsistent geometry or atom mapping, incompatible projector metadata, descriptor nondifferentiability, incompatible model sensitivity, nonfinite intermediates, ill-conditioned response, excessive response residual, unit or shape mismatch, and incomplete force data are hard failures with contextual diagnostics.

Strict DeePHF force evaluation never substitutes `dq_dR_explicit` for a missing or failed `dq_dR_relaxed`.

Strict DeePHF force training, validation, and saved-data testing require both target force and `dq_dR_relaxed` with matching provenance and never degrade silently to energy-only evaluation.

RHF CPHF, UHF UC-PHF, and RKS CPKS convergence are established by independently computed physical response residuals; successful return from a low-level PySCF solver is not sufficient.

The UHF direct response additionally requires independent alpha and beta occupied-space metric, first-order idempotency, particle-number, density-reconstruction, translation, and complete coupled-operator checks.

The RKS direct response additionally requires exact LibXC 7.0.0 and `NumInt` cutoff provenance, prebuilt-grid and canonical radii content, response-generator identity and host-block boundaries, independently finite-differenced translational `w1`, direct Coulomb plus dense `f_xc` operator reconstruction, fixed-grid XC AO-motion reconstruction, separate grid-coordinate and grid-weight response, native gradient grid-response reconstruction, and complete nonorthogonal first-order invariants.

Scalar-adjoint acceptance requires the literal dense transpose residual, an independently applied transpose residual, and an independently applied physical residual after the reference-specific symmetry gate. An RHF, UHF, or RKS adjoint failure never falls back to direct response or an explicit-only correction gradient.

RHF scanner acceptance requires an exact hook-free Mole input or a valid coordinate array, static molecular identity, supported copied SCF controls, fresh SCF convergence, unchanged occupations, continuous occupied-subspace overlap, stable model state during a call, finite energy and gradient, and atomic result publication.

A zero correction must reproduce the native reference energy and gradient, while a coordinate-independent constant correction must retain its constant energy offset and reproduce the native reference gradient and force within the declared numerical tolerance.

The UHF direct and scalar-adjoint gradient drivers clear all public response or adjoint and gradient results at the start of a call and after any failure, so a failed transaction cannot publish stale state.

The RKS direct and scalar-adjoint gradient drivers apply the same fail-closed result publication under explicit backend selection; the scalar-adjoint path rejects scanner and RHF force-data use at their exact API boundaries.

Every failure reports the violated contract category, the relevant reference or tensor identity, and the numerical diagnostic needed to reproduce the decision.

## 11. Numerical acceptance

Explicit descriptor derivatives must agree with fixed-`P` central finite differences, including ordinary atoms, ghost AO centers, and the translational sum rule within the fixture tolerance.

RHF first-order reference densities and UHF alpha, beta, and total first-order reference densities must agree with independently converged displaced-reference finite differences through basis-aware, gauge-invariant comparisons.

Relaxed descriptor derivatives must agree with descriptors from independently converged displaced references over a documented step-size sequence.

RHF, UHF, and strict pure-LDA RKS direct and scalar-adjoint DeePHF total gradients must agree with central finite differences of the complete `e_base + e_corr` energy.

The UHF coupled occupied-virtual operator, per-spin AO metric response, and complete physical response residual must agree with independent AO-integral reconstructions, and omission of either spin, metric, or occupied-virtual contributions must be detectably inconsistent.

UHF explicit, response, and relaxed descriptor derivatives and their additive spin partitions must satisfy their exact sum identities and translational invariants.

RKS first-order densities and relaxed descriptor derivatives must agree with independently converged fresh-grid RKS finite differences over the accepted step sequence, while the native RKS gradient must agree with finite differences of the finite-grid base energy.

The RKS Coulomb and LibXC `f_xc` operator blocks, fixed-grid XC AO motion, separate grid-coordinate and grid-weight Hamiltonian and native-gradient partitions, AO metric response, and complete physical residual must agree with independent AO-integral and dense-grid reconstructions; omission of any one contribution must be detectably inconsistent.

Strict cross-molecule smoke checks validate both `H2/STO-3G` and `LiH/STO-3G` under the same native RKS functional and deterministic grid contract, and the independent grid audit detects response-generator replacement, host-block repartitioning, cached-weight corruption, and translation-preserving `w1` corruption before CPKS.

RHF, UHF, and RKS Z-vector correction partitions must agree with their matching direct-oracle counterparts, and each complete Z-vector gradient must independently agree with central finite differences of `e_base + e_corr`.

RHF scanner results across repeated and displaced geometry sequences must agree with independently constructed fresh methods, while injected SCF, root, model, response, or adjoint failures leave no publishable current result.

Zero and constant corrections must reproduce the complete native RHF, UHF, or grid-response RKS gradient under the corresponding accepted direct or scalar-adjoint contract.

Accepted penalty-free DeePKS total gradients and descriptor values must remain numerically equivalent across the P1 structural refactor.

All numerical tests use deterministic geometries, model parameters, grids where applicable, seeds, double precision, and explicit tolerances.
