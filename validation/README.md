# DeePHF Evaluation and Validation Plan

## Purpose

This directory is the home for release-oriented scientific validation, end-to-end usability evaluation, and performance characterization of the molecular DeePHF implementation. The validation work complements the focused unit and regression tests under `tests/` by exercising larger systems, independent numerical oracles, public interfaces, persisted artifacts, and representative workflows.

The first objective is to validate the exact analytic DeePHF gradients, direct response, Z-vector response, RHF relaxed-force data generation, force-aware training, scanner behavior, and public command-line workflow described by the current project documentation.

## Validation principles

1. Every scientific result must be compared with an independent or differently constructed oracle.
2. Every displaced geometry must use a fresh, tightly converged native PySCF reference unless the test explicitly evaluates scanner cache behavior.
3. A nonzero correction test must demonstrate a material density-response contribution so that an explicit-only implementation cannot pass accidentally.
4. Non-equilibrium, asymmetric geometries must be used so that most Cartesian gradient components are nonzero.
5. All scientific calculations, descriptors, response intermediates, models, and persisted arrays must use double precision.
6. Gradient and force signs, Bohr and Angstrom units, tensor axes, model identity, projector identity, PySCF configuration, and response provenance must be checked at public and persistence boundaries.
7. SCF state continuity, occupations, orbital gaps, descriptor differentiability, response residuals, and operator conditioning must be validated before comparing numerical results.
8. A failed scientific contract must fail closed without publishing a partial gradient, falling back to another backend, or leaving a partial dataset.
9. Performance measurements must use a recorded environment, controlled thread counts, warm-up runs, repeated samples, and machine-readable results.

## Oracle strategy

### Native-reference oracle

For a zero or coordinate-independent constant correction, compare the DeePHF total energy and gradient directly with the native PySCF reference:

```python
e_expected = reference.e_tot
g_expected = reference.nuc_grad_method().kernel()
```

This comparison validates reference delegation, gradient and force signs, atom ordering, units, and the absence of accidental feedback from the correction into the native SCF calculation. It does not exercise the correction response and is therefore only the first validation layer.

### Complete-energy finite-difference oracle

PySCF does not provide a native analytic gradient for the project-specific CorrNet and projected-density correction. The independent oracle for a nonzero correction is a central finite difference of the complete perturbative energy:

```text
E_total(R) = E_reference(P0(R), R) + E_correction(q(P0(R), R))
```

For every positive and negative displacement, construct a fresh molecule, run a fresh PySCF reference to tight convergence, reconstruct the DeePHF method, and evaluate the complete energy. Use displacement sizes `1.0e-3`, `3.0e-4`, and `1.0e-4 Bohr` to demonstrate the finite-difference convergence region and plateau.

### Relaxed-descriptor oracle

Compare the direct-response `dq_dR_relaxed` with central finite differences of descriptors computed after fresh PySCF convergence at every displaced geometry. Also check the exact partition identity:

```text
dq_dR_relaxed = dq_dR_explicit + dq_dR_response
```

The chosen model and geometry must satisfy both of the following anti-vacuity checks:

```text
max(abs(dq_dR_response)) > 1.0e-4 Bohr^-1
explicit-only gradient error > 10 * relaxed-gradient error
```

The numerical thresholds may be tightened after the deterministic validation model is frozen, but they must not be weakened merely to accommodate an unstable electronic state or a descriptor degeneracy.

### Backend-equivalence oracle

Compare the direct and Z-vector correction gradients and total gradients. The Z-vector path must complete one scalar adjoint solve and must not invoke the direct coordinate-response solver. Shared final results alone are insufficient; response and adjoint residuals, reconstruction identities, solve counts, and immutable backend bindings must also be audited.

### Force-training oracle

Use two complementary force-training oracles:

- A deterministic teacher CorrNet supplies exact correction energies and forces for a stable continuous-integration test. A student model exercises data generation, persistence, training, validation, checkpoint restart, saved-data testing, and fresh-geometry inference.
- PySCF RMP2 analytic energies and gradients supply a physically independent target for a slower water-dimer demonstration. The RHF-to-MP2 deltas define `e_corr_target` and `f_corr_target`.

The teacher-student workflow is the deterministic correctness gate. The MP2 workflow evaluates practical usefulness and must report held-out energy and force errors relative to the zero-correction RHF baseline.

## Primary scientific gate: distorted formaldehyde

The first larger oracle system is a non-equilibrium, nonplanar formaldehyde geometry in Angstrom:

```text
C   0.00000000   0.00000000   0.00000000
O   1.25000000   0.08000000  -0.04000000
H  -0.58000000   0.88000000   0.17000000
H  -0.62000000  -0.74000000  -0.28000000
```

Use spherical `6-31G`, disabled symmetry, a converged RHF reference, the established s/p projector basis, and a frozen deterministic nonlinear `tanh` CorrNet. Scale the model so that its correction and response contributions are finite, numerically visible, and small enough to avoid obscuring sign or unit errors with extreme values.

This system has four atoms, twelve Cartesian perturbations, three chemical environments, twenty-two atomic orbitals, occupied-virtual response across s and p spaces, and a stable closed-shell reference. It is materially stronger than a diatomic or a minimal-basis triatomic while remaining suitable for pull-request validation.

An exploratory single-threaded run in the locked environment established feasibility:

- Direct and Z-vector gradients differed by `4.94e-12 Eh/Bohr` at the central geometry.
- A zero-correction direct gradient matched the native PySCF RHF gradient exactly.
- One nonzero-correction Cartesian component differed from a central complete-energy finite difference by `1.32e-8 Eh/Bohr`.
- The direct gradient took approximately `2.24 s`, the Z-vector gradient approximately `1.09 s`, and one positive/negative finite-difference pair approximately `1.29 s` on the exploratory host.

These timings establish feasibility only. They are not portable performance acceptance thresholds.

### Required formaldehyde checks

1. Compare zero-correction direct and Z-vector gradients with the native PySCF RHF gradient.
2. Compare nonzero-correction direct and Z-vector total gradients with complete-energy finite differences for all twelve Cartesian components and all three displacement sizes.
3. Compare the relaxed descriptor Jacobian with finite differences of descriptors after fresh displaced-geometry SCF calculations.
4. Show that the explicit-only descriptor derivative and explicit-only correction gradient fail the complete finite-difference oracle by a material margin.
5. Check direct/Z-vector equivalence for the correction gradient, response gradient, and total gradient separately.
6. Check translational invariance of the total force and rotational covariance of the total energy and gradient.
7. Recompute the same geometry in Bohr and Angstrom and compare energies, descriptors, gradients, and forces.
8. Swap the two hydrogen atom records and verify the corresponding energy invariance and gradient permutation.
9. Compare repeated scanner evaluations with freshly constructed references and methods at the same geometries.
10. Compare Python API, CLI, and persisted NumPy outputs for the same checkpoint, geometry, reference controls, and backend.

### Formaldehyde acceptance criteria

| Quantity | Acceptance criterion |
| --- | --- |
| Zero-correction DeePHF versus native PySCF gradient | Maximum absolute error no greater than `1.0e-10 Eh/Bohr` |
| Direct versus Z-vector total gradient | Maximum absolute error no greater than `1.0e-8 Eh/Bohr` |
| Analytic total gradient versus complete-energy finite difference | Maximum absolute error no greater than `1.0e-5 Eh/Bohr` |
| Relaxed descriptor Jacobian versus finite difference | Maximum absolute error no greater than `1.0e-5 Bohr^-1` |
| Total force sum for RHF | Maximum absolute component no greater than `1.0e-8 Eh/Bohr` |
| Rotation covariance for RHF gradient | Maximum absolute error no greater than `1.0e-8 Eh/Bohr` |
| Direct and adjoint response residuals | No greater than their recorded strict residual tolerances |
| Checkpoint reload | Identical energy and force predictions for the same persisted state |

Relative tolerances must not hide small but scientifically significant absolute errors. The validation report must always include maximum absolute error, root-mean-square error, the worst atom and coordinate, and every finite-difference step.

## Extended reference-family matrix

The extended matrix reuses the same oracle construction while selecting stable systems that exercise each accepted reference family.

| Tier | System | Reference | Basis and functional | Primary purpose |
| --- | --- | --- | --- | --- |
| Pull request | Distorted formaldehyde | RHF | Spherical `6-31G` | Full finite-difference science gate and public workflow |
| Nightly | Distorted formaldehyde | RKS | Spherical `6-31G`, accepted pure-LDA configuration | CPKS and finite-grid coordinate and weight response |
| Nightly | Distorted hydroxymethyl radical | UHF | Spherical `6-31G`, doublet | Coupled alpha/beta response and open-shell state continuity |
| Nightly | Distorted hydroxymethyl radical | UKS | Spherical `6-31G`, doublet, accepted pure-LDA configuration | Spin-coupled CPKS and finite-grid response |
| Release | Asymmetric water dimer | RHF and RMP2 target | Spherical `6-31G` | Force-data production, training, scanner, checkpoint, and held-out inference |

RKS and UKS validation must use the exact public pure-LDA functional normalization, LibXC implementation, deterministic unpruned grid, NumInt settings, and grid-response semantics accepted by the implementation. Every displaced geometry must rebuild the grid according to the same public configuration.

Open-shell validation must additionally compare alpha and beta electron counts, occupations, occupied-subspace overlaps, spin state, and response dimensions across all displaced geometries. A state switch invalidates the finite-difference sample rather than relaxing the numerical tolerance.

## End-to-end water-dimer evaluation

Construct a deterministic set of asymmetric water-dimer geometries spanning intermolecular separation, hydrogen-bond angle, donor and acceptor intramolecular stretches, and small out-of-plane distortions. Keep separate training, validation, and held-out inference geometries.

For each geometry:

1. Run a tightly converged PySCF RHF reference.
2. Run PySCF RMP2 and its analytic nuclear gradient.
3. Define `e_corr_target = E_MP2 - E_RHF`.
4. Define `f_corr_target = F_MP2 - F_RHF`.
5. Generate the model-independent RHF `dq_dR_relaxed` through the direct backend.
6. Persist the strict force dataset and verify every hash, unit, axis convention, reference fingerprint, projector fingerprint, and response diagnostic.
7. Train a smooth double-precision CorrNet with both energy and force loss.
8. Reload the checkpoint, evaluate saved data, and run fresh-geometry direct and Z-vector inference.
9. Compare held-out DeePHF totals with held-out PySCF RMP2 energies and forces.
10. Report errors for the zero-correction RHF baseline, energy-only training, and energy-plus-force training separately.

The deterministic teacher-student version of this workflow must be suitable for automated regression. The RMP2 version may be marked as a slow or release evaluation until its runtime and learning variance are characterized.

## Public-interface and persistence validation

Every scientific system must be evaluated through the concrete Python method, the public factory, the public workflow helper, and the `deepks deephf` command. The following values must agree after accounting for documented shapes:

- Reference, correction, and total energies.
- Descriptor values.
- Direct and Z-vector gradients.
- Forces with the exact negative-gradient convention.
- Convergence state and response diagnostics.
- Persisted `converged.npy`, `e_base.npy`, `e_corr.npy`, `e_tot.npy`, `descriptor.npy`, `gradient.npy`, and `force.npy` arrays.

Scanner validation must include forward and backward geometry sequences so stale cache behavior cannot be hidden by monotonic geometry updates. The final result at a previously visited geometry must match a fresh calculation at that geometry.

## Performance evaluation

Performance evaluation records behavior without weakening scientific checks. Measure the following separately:

- Native PySCF reference wall time.
- Descriptor and correction-energy wall time.
- Direct response construction and solve wall time.
- Z-vector operator construction and solve wall time.
- Complete direct and Z-vector gradient wall time.
- Peak resident memory.
- Response dimension, number of nuclear right-hand sides, solve count, iteration count, and residual.
- Scanner first-frame time and subsequent-frame time.
- Force-data generation time per frame.
- Training time per epoch and inference time per frame.

Use one warm-up followed by at least five measured repetitions for central-geometry inference. Pin BLAS and OpenMP thread counts, record CPU model, logical and physical core counts, available memory, operating system, Python version, NumPy version, PyTorch version, PySCF version, LibXC version, Git revision, and the complete calculation configuration.

Store raw samples as machine-readable data and report median, minimum, maximum, and median absolute deviation. Do not use wall-clock thresholds on shared continuous-integration workers. Performance regression gates may be enabled on a dedicated host after a stable baseline has been collected; the initial recommended alert threshold is a median slowdown greater than 20 percent with unchanged scientific inputs and comparable variability.

The performance study must verify the algorithmic expectations that a scalar Z-vector gradient performs one adjoint solve, direct response handles all coordinate right-hand sides, and repeated scanner use does not reuse stale geometry-dependent intermediates.

## Execution tiers

### Pull-request tier

- Run the existing baseline and focused objective tests.
- Run the distorted-formaldehyde RHF zero-correction native-PySCF comparison.
- Run the distorted-formaldehyde RHF direct/Z-vector comparison.
- Run the complete twelve-component finite-difference oracle with at least two characterized displacement sizes.
- Run unit, sign, permutation, rotation, and scanner freshness checks that do not duplicate expensive displaced calculations.

### Nightly tier

- Run all three formaldehyde finite-difference steps.
- Run RKS, UHF, and UKS direct/Z-vector and finite-difference matrices.
- Run the deterministic water-dimer teacher-student workflow.
- Record performance samples without enforcing a shared-runner timing gate.

### Release tier

- Run the complete locked test suite and build checks.
- Run all scientific validation systems and invariance checks.
- Run the PySCF RMP2 water-dimer training and held-out evaluation.
- Run the dedicated-host performance comparison against the accepted baseline.
- Archive machine-readable inputs, outputs, environment metadata, numerical errors, convergence diagnostics, and timing samples.

## Planned validation artifacts

Validation implementation should keep reusable assets organized under this directory:

```text
validation/
  README.md
  geometries/
  configs/
  scripts/
  baselines/
  reports/
```

Small deterministic geometries and configurations may be versioned. Generated datasets, checkpoints, timing samples, and reports must be written to explicit output directories and must not be committed unless they are intentionally accepted as compact regression fixtures.

Executable pytest regressions derived from this plan belong in objective-specific directories under `tests/`. Validation scripts must call public APIs wherever possible and must keep independent finite-difference and PySCF oracle construction outside production gradient implementations.

## Implementation sequence

1. Freeze the distorted-formaldehyde geometry, projector, deterministic nonlinear model, PySCF controls, state-continuity checks, and expected response-signal lower bound.
2. Implement the independent fresh-reference finite-difference evaluator and machine-readable error report.
3. Add the RHF zero-correction, relaxed-descriptor, direct-gradient, Z-vector, invariance, scanner, and public-workflow validations.
4. Establish the pull-request runtime and decide whether two or three finite-difference steps fit the regular continuous-integration budget.
5. Extend the same harness to the accepted RKS configuration and the selected stable UHF/UKS radical geometry.
6. Add the deterministic water-dimer teacher-student force-training workflow.
7. Add the PySCF RMP2 water-dimer target generator and held-out evaluation.
8. Add controlled performance collection and establish a dedicated-host baseline before enabling regression gates.

Each implementation step must preserve the current focused test suite and must publish enough numerical evidence to identify the worst component, the response contribution, the finite-difference plateau, the electronic-state continuity, and the exact public workflow used to obtain the result.
