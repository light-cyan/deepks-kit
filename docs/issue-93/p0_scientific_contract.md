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
| `O` | `O` | AO-to-projector overlap tensor, partitioned by projector shell. |
| `D` | `D` | Projected-density block `D_s = O_s^T P O_s`. |
| `q` | `q` | Descriptor obtained by concatenating the ascending eigenvalues of every `D_s` block. |
| `dq_dR_explicit` | `dq_dR_explicit` | Nuclear derivative of `q` at fixed numerical AO density matrix `P`, including AO-center and projector-center motion. |
| `dq_dR_response` | `dq_dR_response` | Descriptor derivative caused by the complete first-order reference density, `(partial q / partial P) : P^R`. |
| `dq_dR_relaxed` | `dq_dR_relaxed` | Complete derivative `dq_dR_explicit + dq_dR_response`. |
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

For each projector shell, `O_s[mu,I,m] = <chi_mu | alpha_sIm>`, `D_s[I,m,n] = sum_mu,nu O_s[mu,I,m] P[mu,nu] O_s[nu,I,n]`, and `q` concatenates `eigvalsh(D_s[I])` in configured shell order and ascending eigenvalue order within each block.

The raw-atom-to-descriptor-atom mapping is explicit, stable for a geometry, and stored with results that contain a descriptor atom axis.

## 6. Ghost-center contract

The only canonical ghost predicate is `mol.atom_charge(A) == 0`; element-name prefixes and aliases are not scientific identity tests.

The raw atom axis contains every molecular center, including ghosts, while the descriptor atom axis contains exactly the raw atoms for which `mol.atom_charge(A) != 0`.

AO-center motion is evaluated for every raw atom with an AO slice, including a ghost carrying basis functions.

Projector-center motion is evaluated only for descriptor atoms, using the explicit raw-to-descriptor index mapping.

Elemental correction constants exclude ghosts through the same nuclear-charge predicate.

A ghost coordinate therefore retains a row in every nuclear derivative and force tensor even though the ghost has no descriptor-atom entry of its own.

The shared descriptor accepts a ghost-containing molecular input when its AO slices and descriptor mapping satisfy these rules and the explicit derivative checks pass; the initial DeePHF reference validator admits real-atom RHF references only.

## 7. Tensor axes, signs, dtypes, and units

Runtime tensors omit a frame axis; serialized arrays prepend `n_frame` without changing any remaining axis order.

| Quantity | Runtime axes | Serialized axes | Unit |
|---|---|---|---|
| `P` | `(n_ao, n_ao)` | `(n_frame, n_ao, n_ao)` | Native dimensionless AO density convention. |
| `O_s` | `(n_ao, n_descriptor_atom, n_projector_s)` | Not a required public field. | Dimensionless overlap. |
| `D_s` | `(n_descriptor_atom, n_projector_s, n_projector_s)` | Shell-aware storage only. | Dimensionless. |
| `q` | `(n_descriptor_atom, n_projector)` | `(n_frame, n_descriptor_atom, n_projector)` | Dimensionless. |
| `dq_dR_explicit` | `(n_raw_atom, 3, n_descriptor_atom, n_projector)` | `(n_frame, n_raw_atom, 3, n_descriptor_atom, n_projector)` | Runtime `Bohr^-1`; serialized inverse declared molecular length unit. |
| `dq_dR_response` | `(n_raw_atom, 3, n_descriptor_atom, n_projector)` | Diagnostic storage only. | Runtime `Bohr^-1`; serialized inverse declared molecular length unit. |
| `dq_dR_relaxed` | `(n_raw_atom, 3, n_descriptor_atom, n_projector)` | `(n_frame, n_raw_atom, 3, n_descriptor_atom, n_projector)` | Runtime `Bohr^-1`; serialized inverse declared molecular length unit. |
| `e_base`, `e_corr`, `e_tot`, `e_corr_target` | scalar | `(n_frame, 1)` | `Eh`. |
| `f_reference_variational`, `f_corr_explicit`, `f_corr_explicit_target`, `f_tot` | `(n_raw_atom, 3)` | `(n_frame, n_raw_atom, 3)` | Runtime `Eh/Bohr`; serialized `Eh` per declared molecular length unit. |

Scientific calculations use real double precision; a different dtype is outside strict capability unless independently characterized and declared.

The shared derivative API returns atomic-unit derivatives, while calculation fields convert once to the molecule's declared coordinate unit so stored forces, descriptor Jacobians, and supplied force labels use the same length convention.

For DeePHF, `d(e_corr)/dR[A,x] = sum_I,k (partial e_corr / partial q[I,k]) dq_dR_relaxed[A,x,I,k]`, and the correction-force contribution is the negative of this contraction.

The response component uses a complete basis-aware `P^R`, including occupied-virtual response and occupied-occupied metric response; an occupied-virtual amplitude tensor alone is not `P^R`.

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

## 9. Initial strict RHF capability

The initial DeePHF force capability accepts a finite molecular `pyscf.gto.Mole` reference evaluated by the locked PySCF 2.14 RHF implementation when all of the following conditions hold:

- The reference is a native, undecorated RHF object with real orbitals and integrals.

- The reference is converged, closed shell, integer occupied with values zero or two, and follows a continuous SCF root for the validated nuclear displacements.

- The molecule is nonperiodic, all-electron, real-atom, uses spherical molecular Gaussian AO functions, and has stable atom and AO ordering.

- Density fitting, solvent, QM/MM, external fields, smearing, fractional occupations, symmetry-constrained special occupations, penalty potentials, custom SCF wrappers, and scanner subclasses are absent.

- The correction model produces one real scalar `e_corr`, is evaluated in double precision with fixed projector parameters, and passes the descriptor differentiability and model-compatibility checks.

- The response operator is well conditioned for the configured solver, all response quantities are finite, and the independently calculated residual satisfies the recorded tolerance.

ROHF, UHF, ROKS, RKS, UKS, complex-orbital, periodic, decorated, unconverged, discontinuous-root, and capability-ambiguous references are outside this initial strict RHF set and fail validation before a force calculation begins.

The capability validator checks the concrete reference state and active decorations; a method label such as `xc="HF"` is not sufficient evidence of RHF capability.

## 10. Failure boundaries

Unsupported reference capability, unconverged SCF state, inconsistent geometry or atom mapping, incompatible projector metadata, descriptor nondifferentiability, incompatible model sensitivity, nonfinite intermediates, ill-conditioned response, excessive response residual, unit or shape mismatch, and incomplete force data are hard failures with contextual diagnostics.

Strict DeePHF force evaluation never substitutes `dq_dR_explicit` for a missing or failed `dq_dR_relaxed`.

Strict DeePHF force training, validation, and saved-data testing require both target force and `dq_dR_relaxed` with matching provenance and never degrade silently to energy-only evaluation.

Response convergence is established by an independently computed residual; successful return from a low-level PySCF solver is not sufficient.

A zero or constant correction must reduce `e_tot` and `f_tot` to the native reference values within the declared numerical tolerance.

Every failure reports the violated contract category, the relevant reference or tensor identity, and the numerical diagnostic needed to reproduce the decision.

## 11. Numerical acceptance

Explicit descriptor derivatives must agree with fixed-`P` central finite differences, including ordinary atoms, ghost AO centers, and the translational sum rule within the fixture tolerance.

First-order reference densities must agree with displaced-reference finite differences through a basis-aware, gauge-invariant comparison.

Relaxed descriptor derivatives must agree with descriptors from independently converged displaced references over a documented step-size sequence.

DeePHF total gradients must agree with central finite differences of the complete `e_base + e_corr` energy.

Accepted penalty-free DeePKS total gradients and descriptor values must remain numerically equivalent across the P1 structural refactor.

All numerical tests use deterministic geometries, model parameters, grids where applicable, seeds, double precision, and explicit tolerances.
