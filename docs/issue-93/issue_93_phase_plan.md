# Issue #93 Implementation Phase Plan

**Objective:** Complete [deepmodeling/deepks-kit Issue #93](https://github.com/deepmodeling/deepks-kit/issues/93) in the `light-cyan/deepks-kit` fork by implementing exact analytic DeePHF nuclear forces and force-aware training within an explicit support domain.

**Plan status:** Working outline

**Current phase:** P0 has started through the `uv` setup migration, dependency lock, and baseline smoke tests at `b1a54287489b39f31e657ca38e5ef4e2978b9056`.

**Assessment basis:** [2026-08-19 technical assessment](./deepks_issue_93_assessment.0819.md)

## 1. Delivery principles

- Treat Issue #93 as an umbrella objective delivered through reviewable phases and independently verifiable stage gates.

- Establish one scientifically correct RHF direct backend before optimizing inference or expanding reference types.

- Preserve the legacy meaning of `grad_vx` and all existing DeePKS inputs unless a user explicitly selects the new DeePHF mode.

- Use a minimal non-self-consistent DeePHF energy method and central finite differences as the total-energy oracle.

- Reuse existing descriptor primitives only after characterization tests establish their signs, shapes, units, occupation factors, spin conventions, ghost behavior, and differentiability domain.

- Require each supported reference class to pass direct-response validation before accepting its corresponding Z-vector backend.

- Fail explicitly for unsupported references, failed response solves, incompatible data, incompatible models, and nondifferentiable descriptor cases.

- Keep scientific correctness milestones separate from large-data performance and broad compatibility expansion.

- Place analytic-force tests under `tests/analytic_forces/` and run that objective directory before the complete suite.

## 2. Scope

### 2.1 Required end state

- A dedicated non-self-consistent molecular DeePHF Python API exists.

- Supported RHF, UHF, RKS, and UKS references provide validated analytic DeePHF correction and total gradients.

- Direct and Z-vector backends agree within method-specific tolerances.

- Relaxed descriptor derivatives have versioned semantics and provenance.

- Force-aware training, validation, saved-data testing, checkpoint reload, and direct molecular inference are available.

- Existing DeePKS workflows and legacy explicit `grad_vx` datasets retain their current meaning.

- Geometry scanner usage invalidates every geometry-dependent cache correctly.

- CI exercises analytic-force and force-training regression tests in a declared dependency environment.

### 2.2 Initial exclusions

- Periodic and ABACUS response theory

- Analytic DeePHF Hessians

- Nonadiabatic derivatives

- Differentiation through the reference SCF procedure with PyTorch

- ROHF and ROKS

- Fractional occupations and smearing

- Complex, noncollinear, and spinor orbitals

- Unverified density-fitting, solvent, QM/MM, external-field, ECP, and custom-wrapper combinations

## 3. Dependency graph

```text
P0 Contracts, characterization tests, and finite-difference infrastructure
 |
 v
P1 Shared descriptor context, minimal DeePHF method, and total-energy oracle
 |
 v
P2 RHF direct response
 |-------------------------|-------------------------|-------------------------|
 v                         v                         v                         v
P3A Data and training      P3B RHF Z-vector         P3C UHF direct           P3D RKS direct
                                                     then UHF Z               then RKS Z
                                                       |                         |
                                                       |-----------|-------------|
                                                                   v
                                                            P4 UKS direct
                                                            then UKS Z
                                                                   |
                                                                   v
                                             P5 Scanner, workflow integration, storage,
                                                performance, compatibility, and docs
```

P3A, P3B, P3C, and P3D share P2 as a dependency and may progress in parallel when they do not edit the same interfaces.

P4 depends on stable unrestricted spin conventions from P3C and stable KS grid and XC conventions from P3D.

P5 consolidates interfaces after the scientific backends and schema have passed their stage gates.

## 4. P0 — Contracts, characterization, and reproducibility

### 4.1 Goals

- Freeze the initial scientific and software support contract before changing response-sensitive code.

- Characterize the current descriptor, gradient, model, data, and scanner behavior.

- Establish deterministic finite-difference utilities and regression fixtures.

### 4.2 Tasks

- [ ] Record the implementation repository, upstream issue, target base commit, active branch, and intended upstream synchronization procedure.

- [ ] Use Python 3.11 and the locked PySCF 2.14, Torch 2.13, and NumPy environment as the first supported development combination.

- [ ] Add a CI job that uses `uv sync --locked --python 3.11` and runs the baseline and analytic-force objective suites.

- [ ] Define gradient and force signs at every API and data boundary.

- [ ] Define coordinate units, conversion rules, atom ordering, raw-atom versus descriptor-atom axes, `atmlst` behavior, and ghost-center semantics.

- [ ] Define the version-one response-data schema, including names for explicit, response, and relaxed descriptor derivatives.

- [ ] Define required dataset provenance: reference class, XC, basis, ECP, charge, spin, occupations, projector hash, geometry, software versions, SCF controls, CP controls, grid settings, backend, and differentiability diagnostics.

- [ ] Define the initial reference support contract: converged real RHF, integer occupations, continuous root, sufficiently conditioned response, compatible descriptor and model, no penalty, and no unsupported decorator.

- [ ] Define a capability-error taxonomy for unsupported reference classes, modifiers, occupations, roots, grids, ghost cases, models, and descriptor differentiability.

- [ ] Define the descriptor-degeneracy decision procedure, distinguishing structural fixed-rank zero spaces, symmetry-induced splitting, near crossings, and nondifferentiable ordered-eigenvalue behavior.

- [ ] Define model compatibility checks for shell-wise normalization, linear bypass weights, degenerate-subspace sensitivities, activation smoothness, projector identity, and checkpoint provenance.

- [ ] Add deterministic central-difference helpers that rerun the reference calculation and evaluate the complete perturbative `E_ref + E_corr` energy.

- [ ] Add a fixed-AO-density finite-difference test for the existing explicit projected-density and eigenvalue derivatives.

- [ ] Add a finite-difference test for existing `t_make_grad_eig_dm` or an equivalent density-direction derivative.

- [ ] Add convention tests for `t_ele_grad` and `make_grad_eig_egrad` in restricted and unrestricted orbital spaces.

- [ ] Add a nonzero, smooth, penalty-free DeePKS analytic-gradient regression against total-energy finite differences.

- [ ] Add tests that expose the current `X-H` ghost AO-center omission and the `GHOST-H` classification inconsistency.

- [ ] Add tests showing that strict force training fails when a force label, relaxed Jacobian, metadata record, or compatible schema is missing.

- [ ] Select low-symmetry C1 molecular fixtures for primary derivative tests and separate fixtures for structural degeneracy, symmetry splitting, near crossing, open shell, and DFT grid convergence.

### 4.3 Deliverables

- A documented version-one scientific support contract

- A versioned data-schema specification

- Deterministic finite-difference test helpers

- Characterization coverage for the existing descriptor and DeePKS gradient behavior

- CI for the locked development environment

### 4.4 Exit gate

- Current supported DeePKS behavior remains unchanged within explicit tolerances.

- Existing `grad_vx` is permanently identified as the legacy explicit fixed-density derivative.

- The initial exact-DeePHF capability boundary has a deterministic accept-or-error outcome.

- Every response-sensitive helper that will be reused has a numerical convention test.

- The full baseline and P0 analytic-force characterization suites pass in the locked environment.

## 5. P1 — Shared descriptor context and minimal DeePHF method

### 5.1 Dependencies

- P0 exit gate

### 5.2 Goals

- Separate reusable descriptor evaluation from the self-consistent `DSCF` and `UDSCF` mixins.

- Establish a non-self-consistent DeePHF energy object that cannot feed the learned potential into reference SCF iterations.

- Provide the total-energy oracle required by every analytic-gradient stage.

### 5.3 Tasks

- [ ] Introduce a shared descriptor context for projector construction, overlap integrals, projected densities, shell eigenvalues, `dq/dP`, explicit `dD/dR`, explicit `dq/dR`, and geometry-local cache invalidation.

- [ ] Make existing `NetMixin` and `NetGradMixin` delegate to the shared context while preserving public behavior.

- [ ] Replace string-prefix ghost detection with a canonical predicate based on nuclear charge or normalized PySCF atom identity.

- [ ] Separate raw AO centers from descriptor projector centers so a ghost AO center can move without creating a learned projector center.

- [ ] Implement the selected Phase 0 descriptor-degeneracy and model-compatibility validators.

- [ ] Add a minimal `DeePHF` object composed around a converged native PySCF mean-field object.

- [ ] Expose `e_ref`, `e_corr`, `e_tot`, delegated `make_rdm1()`, descriptor evaluation, correction-potential evaluation, and capability diagnostics.

- [ ] Verify that evaluating the DeePHF correction never modifies the reference Fock iterations, orbitals, occupations, or density.

- [ ] Add a geometry refresh path that invalidates projector, overlap, descriptor, and method caches.

- [ ] Add the complete perturbative-energy central-difference oracle to the method test support.

- [ ] Reject unconverged references, penalty-bearing references, and unsupported wrappers before energy or gradient evaluation.

### 5.4 Deliverables

- Shared descriptor context

- Minimal non-self-consistent DeePHF energy API

- Corrected and tested ghost-center handling or an explicit temporary rejection boundary

- Complete perturbative-energy finite-difference oracle

### 5.5 Exit gate

- `e_tot` equals the native reference energy plus the model correction evaluated on the unchanged reference density.

- Reference orbitals, occupations, and density are identical before and after correction evaluation.

- Existing DeePKS descriptor and gradient results remain unchanged within characterization tolerances.

- Geometry changes refresh all descriptor-side caches.

- Supported ghost behavior matches fixed-density finite differences, or unsupported ghost inputs fail before evaluation.

## 6. P2 — RHF direct-response scientific backend

### 6.1 Dependencies

- P1 exit gate

### 6.2 Goals

- Produce the complete model-independent RHF relaxed descriptor derivative.

- Establish the scientific oracle for force training, Z-vector work, and later reference types.

### 6.3 Tasks

- [ ] Add an RHF response adapter around matching PySCF Hessian nuclear perturbation and overlap-aware CPHF helpers.

- [ ] Construct nuclear first-order Fock or core matrices and overlap perturbations with explicit atom and coordinate batching.

- [ ] Obtain full first-order occupied MO coefficients and retain the occupied-occupied orthonormality or metric contribution.

- [ ] Construct the full AO first-order density `P^R` with the correct restricted occupation factor.

- [ ] Reapply the CP equations after the solve and calculate a normalized residual because the low-level solver does not provide a universal convergence object.

- [ ] Record CP tolerance, cycle limit, level shift, residual, batching, and failure status.

- [ ] Form relaxed projected-density derivatives before applying the selected eigenvalue differentiability contract.

- [ ] Form `dq_dR_response = q_P : P^R` and `dq_dR_relaxed = dq_dR_explicit + dq_dR_response`.

- [ ] Support `atmlst` and coordinate batching without changing returned atom-axis semantics.

- [ ] Add capability checks for occupations, real orbitals, response conditioning, root continuity, symmetry behavior, ghost policy, and unsupported reference decorators.

- [ ] Compare `P^R` against independently converged displaced RHF densities with a basis-aware and gauge-invariant metric.

- [ ] Compare explicit derivatives against fixed-density finite differences.

- [ ] Compare relaxed projected-density and descriptor derivatives against independently converged displaced-reference finite differences over multiple step sizes.

- [ ] Compare the direct total DeePHF gradient against central differences of `E_ref + E_corr`.

- [ ] Verify that zero and constant correction models reproduce the native RHF gradient.

- [ ] Add translation, rotation, atom-subset, ghost-policy, and repeated-geometry regression cases.

### 6.4 Deliverables

- RHF direct response adapter

- Complete `P^R` and relaxed projected-density derivative

- Versioned explicit, response, and relaxed descriptor derivatives

- CP residual and diagnostic interface

- RHF direct analytic DeePHF gradient

### 6.5 Exit gate

- Explicit, response, and relaxed components each pass their designated finite-difference comparison.

- The complete RHF DeePHF gradient reaches `1e-5 Eh/Bohr` or a tighter fixture-specific tolerance and shows the expected finite-difference plateau with at least two step sizes.

- CP residual failures and unsupported cases produce hard errors.

- Zero correction agrees with the native RHF gradient to numerical precision.

- The P2 analytic-force objective suite passes deterministically in double precision.

## 7. P3A — RHF force-aware data, training, validation, and testing

### 7.1 Dependencies

- P2 exit gate

### 7.2 Parallelism

- May proceed in parallel with P3B, P3C, and P3D after shared P2 interfaces are frozen.

### 7.3 Goals

- Deliver the first end-to-end user-visible objective of Issue #93 without waiting for Z-vector optimization.

### 7.4 Tasks

- [ ] Implement the version-one explicit, response, and relaxed derivative fields without changing legacy `grad_vx` semantics.

- [ ] Write and validate per-system response metadata.

- [ ] Make strict DeePHF force mode fail when relaxed derivatives, labels, metadata, units, projector identity, or compatible schema are missing.

- [ ] Validate schema consistency across every system in a grouped reader.

- [ ] Centralize energy and force prediction in a helper shared by training, validation, and saved-data testing.

- [ ] Report energy and force losses and metrics separately.

- [ ] Add force-only metrics that do not require high-level reference forces during unlabeled inference.

- [ ] Retain existing legacy and DeePKS force-training behavior behind explicit modes.

- [ ] Add parameter finite-difference tests for gradients of the force loss.

- [ ] Add checkpoint save, reload, and deterministic energy-and-force reproduction tests.

- [ ] Add a compact RHF energy-and-force training example and fixture.

- [ ] Add a minimal explicit `base | deephf | deepks` configuration boundary while preserving the old meaning of `deepks scf -m model.pth`.

- [ ] Set an explicit small-system limit for the eager NumPy implementation until chunked storage is delivered.

### 7.5 Deliverables

- Strict response-data schema and reader

- Shared energy and force prediction helper

- RHF exact-DeePHF force loss, validation metrics, and saved-data test path

- End-to-end RHF training example

- Minimal explicit calculation-mode integration

### 7.6 Exit gate

- Legacy explicit Jacobians cannot silently enter exact DeePHF force mode.

- Force-loss parameter gradients match selected-parameter finite differences.

- Training and validation report independent energy and force metrics.

- A checkpoint reload reproduces energy and force predictions.

- A small RHF force-trained model completes the end-to-end data, training, validation, and test workflow.

## 8. P3B — RHF Z-vector inference

### 8.1 Dependencies

- P2 exit gate

### 8.2 Parallelism

- May proceed in parallel with P3A, P3C, and P3D.

### 8.3 Goals

- Replace coordinate-count-scaled direct response with one verified adjoint solve for scalar model inference.

### 8.4 Tasks

- [ ] Derive the correction orbital RHS from the AO correction potential using the same restricted occupation and orbital-rotation convention as the direct backend.

- [ ] Reuse or characterize the existing occupied-virtual transformation helper instead of duplicating convention-sensitive factors.

- [ ] Implement the actual transpose response action used by the chosen PySCF and preconditioning convention.

- [ ] Derive and implement the correction-specific AO metric or overlap contraction.

- [ ] Share nuclear perturbation RHS construction, signs, occupations, and atom batching with the direct backend.

- [ ] Calculate and record an adjoint residual.

- [ ] Compare explicit projector, AO metric, and orbital-response contributions separately against the direct decomposition.

- [ ] Compare total Z-vector gradients against direct gradients and total-energy finite differences.

- [ ] Implement `nuc_grad_method(response_method="direct" | "zvector")`.

### 8.5 Deliverables

- RHF Z-vector backend

- Correction-specific AO metric implementation

- Direct and Z backend selection

### 8.6 Exit gate

- Every Z-vector contribution agrees with its direct counterpart within an explicit internal tolerance.

- Total direct and Z-vector gradients agree to a tight RHF double-precision target.

- Z-vector results agree with total-energy finite differences.

- The Z backend becomes the default only after all equivalence tests pass.

## 9. P3C — UHF direct response and UHF Z-vector

### 9.1 Dependencies

- P2 exit gate

### 9.2 Parallelism

- May proceed in parallel with P3A, P3B, and P3D.

### 9.3 Tasks

- [ ] Stop `build_mol` from replacing an explicitly supplied user spin.

- [ ] Define and preserve the current spin-summed descriptor contract `P_alpha + P_beta`.

- [ ] Implement coupled UCPHF first-order alpha and beta orbitals and densities.

- [ ] Include spin-resolved occupied-occupied metric contributions.

- [ ] Validate `P_alpha^R`, `P_beta^R`, their sum, and the relaxed descriptor response independently.

- [ ] Verify that the same correction potential acts in both spin channels under the current descriptor definition.

- [ ] Add well-behaved open-shell, nondegenerate, integer-occupation fixtures.

- [ ] Reject ROHF, fractional occupations, unsupported spin decorators, root changes, and ill-conditioned response cases.

- [ ] Implement UHF Z-vector only after UHF direct passes.

- [ ] Compare UHF direct, UHF Z, and total-energy finite differences.

### 9.4 Deliverables

- General-spin runner behavior

- UHF direct relaxed descriptor response

- UHF direct and Z-vector DeePHF gradients

### 9.5 Exit gate

- Spin-resolved first-order densities and their sum pass finite-difference validation.

- UHF direct total gradients pass total-energy finite differences.

- UHF Z-vector results pass direct equivalence.

- Unsupported open-shell reference types fail deterministically.

## 10. P3D — RKS direct response and RKS Z-vector

### 10.1 Dependencies

- P2 exit gate

### 10.2 Parallelism

- May proceed in parallel with P3A, P3B, and P3C.

### 10.3 Tasks

- [ ] Define the first RKS functional tier, beginning with one deterministic conventional LDA or GGA configuration.

- [ ] Define whether each tier targets a continuous quadrature limit or the derivative of a specified discretized grid.

- [ ] Reuse the matching PySCF RKS Hessian nuclear RHS rather than treating `mf.gen_response` as the entire nuclear perturbation.

- [ ] Record grid level, atom grids, pruning, grid rebuild policy, grid-response setting, and functional identity in metadata.

- [ ] Validate the XC response kernel and nuclear RHS separately where practical.

- [ ] Run dense-grid and displacement-step convergence studies.

- [ ] Add LDA, GGA, and global-hybrid tiers one at a time.

- [ ] Keep meta-GGA, NLC, range-separated hybrid, custom functional, and unsupported grid configurations outside a tier until independently validated.

- [ ] Implement RKS Z-vector only after the matching RKS direct tier passes.

- [ ] Compare RKS direct, RKS Z, and consistent-grid total-energy finite differences.

### 10.4 Deliverables

- Versioned RKS capability matrix

- RKS direct response for accepted XC tiers

- RKS Z-vector response for accepted XC tiers

- Grid and pruning provenance

### 10.5 Exit gate

- Each supported XC tier has matching direct, Z-vector, and finite-difference evidence.

- Grid and displacement convergence plateaus are documented by tests.

- Unsupported XC and grid configurations fail capability checks.

- No RKS tier is generalized to UKS until the P4 spin and cross-spin requirements pass.

## 11. P4 — UKS direct response and UKS Z-vector

### 11.1 Dependencies

- P3C UHF direct exit gate

- P3D RKS direct exit gate

### 11.2 Goals

- Combine unrestricted spin response with tested KS nuclear RHS and grid semantics.

### 11.3 Tasks

- [ ] Implement coupled alpha and beta CPKS response including cross-spin XC kernels.

- [ ] Preserve the spin-summed descriptor response while retaining spin-resolved diagnostics.

- [ ] Reuse the accepted RKS functional tiers and grid contracts rather than creating a separate implicit support surface.

- [ ] Validate spin-resolved first-order densities and the summed descriptor response.

- [ ] Add open-shell UKS fixtures for each accepted XC tier.

- [ ] Implement UKS Z-vector only after UKS direct passes.

- [ ] Compare UKS direct, UKS Z, and consistent-grid total-energy finite differences.

### 11.4 Deliverables

- UKS direct response for accepted XC tiers

- UKS Z-vector response for accepted XC tiers

- Spin-resolved and grid-resolved diagnostics

### 11.5 Exit gate

- UKS spin-resolved response passes finite differences.

- UKS total direct gradients pass consistent-grid total-energy finite differences.

- UKS Z-vector results pass direct equivalence.

- Unsupported functionals, occupations, roots, and decorators fail deterministically.

## 12. P5 — Scanner, full integration, storage, performance, and documentation

### 12.1 Dependencies

- P3A exit gate

- P3B exit gate

- Accepted reference backends from P3C, P3D, and P4

### 12.2 Tasks

- [ ] Finalize explicit `base | deephf | deepks` CLI and configuration behavior.

- [ ] Implement `as_scanner()` so every geometry refreshes the native reference SCF, projector integrals, descriptors, grids when relevant, response intermediates, and model-side geometry state.

- [ ] Add repeated-call, A-to-B-to-A geometry, geometry-optimizer, stale-cache, failed-reference, model-change, grid-change, and backend-switching tests.

- [ ] Integrate supported DeePHF data generation and inference into workflow and statistics paths without changing legacy defaults.

- [ ] Add chunked or lazy response-Jacobian storage and loading.

- [ ] Define precision, compression, chunking, shuffling, and multi-system batching policies.

- [ ] Evaluate Jacobian-vector or vector-Jacobian alternatives for systems where full quadratic storage is impractical.

- [ ] Benchmark direct and Z-vector backends by atom count, basis size, descriptor size, reference type, XC tier, memory, and wall time.

- [ ] Add geometry-optimizer examples using the supported scanner.

- [ ] Add migration tooling or documented conversion procedures for versioned data without reinterpreting legacy `grad_vx`.

- [ ] Expand the supported PySCF version matrix only after the compatibility adapter passes version-specific tests.

- [ ] Add complete user documentation for theory, modes, signs, units, schema, provenance, convergence, capability errors, supported references, grids, scanner use, and performance.

- [ ] Add developer documentation for response conventions, residual calculation, direct and adjoint operators, cache ownership, and extension requirements.

- [ ] Run the analytic-force objective suite, full suite, locked dependency synchronization, and build verification.

### 12.3 Deliverables

- Complete user-facing CLI and workflow integration

- Correct scanner and cache lifecycle across every accepted backend

- Scalable response-data path

- Direct-versus-Z performance report

- Supported dependency and reference matrix

- User and developer documentation

### 12.4 Exit gate

- Every advertised mode, reference type, XC tier, backend, and data format has a passing regression test.

- Legacy inputs preserve their documented behavior.

- Large response datasets can be trained without eager loading of the complete corpus.

- Scanner and geometry-optimizer examples run reproducibly.

- Repeated scanner calls show no stale reference, projector, descriptor, grid, model, or response state.

- `uv run pytest tests/analytic_forces`, `uv run pytest`, `uv sync --locked --python 3.11`, and `uv build` pass.

## 13. Milestones

| Milestone | Completed phases | User-visible outcome |
|---|---|---|
| M0 Reproducible foundation | P0 | Locked environment, contracts, characterization tests, and CI |
| M1 RHF scientific oracle | P1 and P2 | Exact RHF direct analytic DeePHF gradient with finite-difference evidence |
| M2 RHF force-aware training | P3A | End-to-end relaxed-Jacobian data, force training, validation, and saved-data testing |
| M3 RHF optimized inference | P3B | Verified RHF Z-vector backend |
| M4 Open-shell and DFT tiers | P3C, P3D, and P4 | Validated UHF, RKS, and UKS support by declared capability tier |
| M5 Production integration | P5 | Scanner, complete workflow, scalable storage, compatibility matrix, benchmarks, and documentation |

## 14. Re-estimation points

- Re-estimate remaining work after M1 because RHF direct will expose the actual PySCF convention, residual, degeneracy, and cache complexity.

- Re-estimate remaining work after M3 because the RHF adjoint implementation will expose the actual transpose and AO metric complexity.

- Estimate each XC and unrestricted support tier independently rather than maintaining one aggregate four-reference schedule.

## 15. Review boundaries

- Keep descriptor refactoring separate from response-theory changes whenever possible.

- Keep direct and Z-vector implementations in separate review units.

- Keep schema and reader changes reviewable independently from scientific response code.

- Keep each new reference class and each XC tier behind its own capability tests.

- Do not set a backend as default or advertise a support tier until its exit gate passes.
