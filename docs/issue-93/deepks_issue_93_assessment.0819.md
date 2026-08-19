# Updated Technical Assessment of deepks-kit Issue #93

**Issue:** [deepmodeling/deepks-kit Issue #93](https://github.com/deepmodeling/deepks-kit/issues/93)

**Implementation repository:** [light-cyan/deepks-kit](https://github.com/light-cyan/deepks-kit)

**Repository snapshot:** `b1a54287489b39f31e657ca38e5ef4e2978b9056`

**Assessment date:** 2026-08-19

**Related assessment:** [2026-08-18 assessment](./deepks_issue_93_assessment.0818.md)

## 1. Executive judgment

Issue #93 is technically achievable in this fork, and an RHF direct-response implementation is demonstrably feasible with the currently locked PySCF 2.14 environment.

The full issue is not a local patch. It combines response theory, descriptor differentiability, a non-self-consistent method API, data-schema evolution, force-aware training, multiple reference classes, scanner lifecycle management, and PySCF compatibility work.

The scientifically defensible target is the exact first analytic derivative of the defined approximate DeePHF energy within a documented support domain. It is not an exact physical force and cannot be promised for arbitrary PySCF objects, occupations, exchange-correlation functionals, grids, descriptor degeneracies, or decorated references.

The core diagnosis in Issue #93 and the 2026-08-18 assessment is reliable: the current `grad_vx` is an explicit descriptor derivative at fixed AO density, whereas perturbative DeePHF requires the response of the converged reference density to nuclear displacement.

The 2026-08-18 assessment is suitable as design-review input, but its implementation sequence, several reliability claims, and parts of its issue characterization require the corrections recorded below.

## 2. Repository and project context

The `light-cyan/deepks-kit` fork exists to complete upstream Issue #93. The local `codex/DeePHF-analytic-forces` branch currently contains the setup migration at `b1a5428`: Python 3.10+ metadata, a Python 3.11 development standard, `uv` dependency locking, PySCF and Torch dependency declarations, and baseline smoke tests.

This setup work is Phase 0 infrastructure for Issue #93 rather than an unrelated change. The scientific SCF, descriptor, model, and gradient implementation remains materially unchanged from upstream commit `4f133fb`.

The upstream issue remains an unreviewed design proposal: it is open, has no public comments, assignee, label, milestone, linked branch, or linked pull request, and its body states that it was generated with Codex. This provenance does not invalidate the mathematics, but upstream acceptance and technical correctness are separate questions.

The implementation target is now clear: develop the feature in this fork against the upstream Issue #93 objective. Any later upstream contribution still requires maintainer review.

## 3. Verified current-code diagnosis

The projected-density descriptor is built from `D = O^T P O` and shell eigenvalues in [`deepks/scf/scf.py`](../../deepks/scf/scf.py#L29-L50).

The existing `t_make_grad_pdm_x` differentiates molecular-AO/projector overlaps and projector-center motion while treating the supplied AO density matrix as fixed; `t_make_grad_eig_x` then contracts that explicit derivative with an eigenvalue Jacobian in [`deepks/scf/grad.py`](../../deepks/scf/grad.py#L41-L73).

No CPHF, UCPHF, CPKS, first-order reference-density builder, or Z-vector implementation exists in the repository.

For perturbative DeePHF, the required descriptor derivative is `J_relaxed = J_explicit + q_P : P0^R`. The missing term is generally nonzero because the correction energy does not participate in the reference HF/KS stationarity equations.

For ordinary converged, differentiable, penalty-free self-consistent DeePKS calculations, the learned potential participates in the SCF equations and the full corrected functional is stationary. The inherited PySCF gradient plus the explicit correction derivative can therefore produce the first derivative without a separate orbital-response solve.

That DeePKS statement does not cover the repository's penalty path. Penalty potentials enter the Fock matrix through [`deepks/scf/penalty.py`](../../deepks/scf/penalty.py#L25-L41), but the reported energy does not include matching penalty energy and gradient hooks. A strict DeePHF capability layer should reject penalty-bearing or otherwise decorated reference objects until their derivative semantics are implemented and tested.

## 4. Existing primitives that should be reused

The repository already contains `t_make_grad_eig_dm`, which computes `q_P = partial q / partial P_AO`, in [`deepks/scf/scf.py`](../../deepks/scf/scf.py#L76-L85). The RHF direct response contribution can therefore be formed by contracting this tensor with a complete first-order AO density.

The repository also contains `t_ele_grad` and `make_grad_eig_egrad` in [`deepks/scf/addons.py`](../../deepks/scf/addons.py#L12-L35). These functions transform AO descriptor sensitivities into occupied-virtual orbital-gradient coordinates and may provide a starting point for correction RHS construction after restricted, unrestricted, occupation-factor, and sign conventions are characterized numerically.

The current force-training expression already evaluates `-J * dE/dq` with `create_graph=True` in [`deepks/model/train.py`](../../deepks/model/train.py#L107-L138). Force-aware DeePHF training does not require differentiating through PySCF once a verified model-independent relaxed Jacobian is available.

The current field layer already exposes projected density and explicit projected-density derivatives through `proj_dm` and `grad_dmx` in [`deepks/scf/fields.py`](../../deepks/scf/fields.py#L63-L70) and [`deepks/scf/fields.py`](../../deepks/scf/fields.py#L128-L135). A direct implementation can use relaxed projected-density derivatives as an internal oracle before committing every public interface to individual eigenvalue derivatives.

These primitives reduce the RHF direct scope. They should receive characterization tests before extraction or refactoring rather than being rewritten from scratch.

The current package layout should not become the target architecture. [`deepks/scf/scf.py`](../../deepks/scf/scf.py) mixes method-independent descriptor operations with self-consistent DeePKS classes, [`deepks/scf/grad.py`](../../deepks/scf/grad.py) mixes local descriptor derivatives with DeePKS variational gradients, and [`deepks/scf/fields.py`](../../deepks/scf/fields.py) couples method results to data and training concerns. This structure creates semantic ambiguity that cannot be solved by renaming individual variables in place.

The implementation should extract a method-independent descriptor package and place self-consistent DeePKS and perturbative DeePHF in separate sibling method packages. The method packages should depend on the shared descriptor and model contracts but never on one another; data, training, CLI, and workflows should depend on stable method outputs rather than method internals. Because this fork has no compatibility requirement, the refactor should preserve validated numerical behavior instead of old import paths, field names, aliases, or configuration forms.

## 5. Numerical feasibility evidence

The response-theory feasibility checks were run in memory against `b1a5428` with Python 3.11.15, PySCF 2.14.0, Torch 2.13.0, NumPy 2.4.6, double precision, converged references, and central finite differences. The current branch also carries deterministic characterization tests for the existing DeePKS and example paths.

| Check | Observation |
|---|---|
| Existing explicit descriptor derivative versus fixed-AO-density finite difference | Agreement at approximately `1e-9` to `1e-7` depending on the fixture and projector |
| Existing explicit derivative versus displaced-reference relaxed derivative | Missing response contribution is large; a distorted-water check differed by up to `5.76e-1` |
| RHF direct relaxed Jacobian built from PySCF `make_h1`, full `mo1`, complete `P^R`, and existing `q_P` | Maximum error versus displaced-SCF descriptor finite difference: `3.30e-9` |
| Total gradient of `E_ref + E_corr` using the RHF direct relaxed Jacobian | Maximum error versus total-energy finite difference: `1.42e-9 Eh/Bohr` |
| Existing nonzero, smooth, penalty-free self-consistent DeePKS gradient | Agreement with total-energy finite difference at approximately `1e-9` to `1e-8 Eh/Bohr` |
| Existing RKS and UKS DeePKS gradients on a deterministic unpruned grid | Enabling PySCF grid response gives agreement with total-energy finite differences within approximately `1.6e-8 Eh/Bohr`; the default disabled grid response misses terms of approximately `1.2e-2` to `1.5e-2 Eh/Bohr` on the fixture |
| Current complete test suite | `42 passed`, including descriptor Jacobians, nonzero RHF/UHF/RKS/UKS DeePKS gradients, scanner behavior, example assets and configuration, force-data contraction, training, checkpoint reload, and model-driven SCF |

These checks prove that the RHF direct architecture has a working implementation path in the current dependency environment and establish a refactor baseline for accepted existing paths. They do not validate a production response residual, Z-vector transpose action, DeePHF response for UHF, RKS, or UKS, broad PySCF compatibility, or production response-data storage.

The original zero-correction smoke test remains in [`tests/baseline/test_scf_grad.py`](../../tests/baseline/test_scf_grad.py#L10-L36), while the new characterization suite covers nonzero existing DeePKS behavior and finite-difference agreement. These tests protect the refactor but do not validate the missing perturbative DeePHF response.

## 6. Reliability of Issue #93

### 6.1 Reliable claims

- Perturbative DeePHF needs reference orbital or density response because the learned correction does not enter the reference SCF stationarity equations.

- Direct CP-HF or CPKS can generate a model-independent relaxed descriptor Jacobian for data generation and force-aware training.

- A scalar Z-vector or adjoint formulation is the appropriate inference optimization after a direct backend exists as an oracle.

- A dedicated non-self-consistent method object should compose around a native converged PySCF reference instead of reusing `DSCF` or `UDSCF`.

- Existing fixed-density Jacobian data cannot supply the missing response term and must be rejected or regenerated when the force-data schema is revised.

- Explicit calculation modes, force evaluation, tests, provenance, scanner support, and documentation are needed.

### 6.2 Claims that need narrowing or amendment

- `RHF/UHF/RKS/UKS` cannot mean every object accepted by PySCF or every `xc != HF` configuration. Initial support needs an explicit capability matrix.

- The direct backend design already mentions overlap and orthonormality response, so the issue is not missing that direct concept. The implementation still needs complete `mo1` reconstruction and convention tests.

- The Z-vector design omits a correction-specific AO metric or overlap contraction that is independent of the occupied-virtual adjoint term. Direct and Z results are not equivalent until this contribution is included.

- The issue recognizes descriptor degeneracy and requests a subspace-safe treatment before exact support is claimed. Its missing element is a precise acceptance, rejection, and model-compatibility contract, not awareness of the problem.

- The issue already contains Phases 0 through 5 and requests reviewable pull requests. It needs a reordered dependency graph and explicit stage gates rather than the addition of a staged plan from nothing.

- The issue already points direct CPKS toward matching PySCF Hessian nuclear perturbations. The missing KS specification is the discrete grid, pruning, grid-response, supported-functional, and capability contract.

- The proposed full-scope estimate of three to five developer weeks has no implementation or public-review evidence and should not be used as a delivery commitment.

## 7. Reliability and corrections for the 2026-08-18 assessment

### 7.1 Findings that remain reliable

- The central explicit-versus-relaxed diagnosis is correct.

- Complete first-order MO coefficients, including the occupied-occupied metric response, are required to construct a direct `P^R`.

- PySCF's low-level CPHF interfaces do not provide a universal convergence object, so the implementation must calculate and record a response residual.

- The Z-vector implementation needs a verified transpose convention and an explicit or equivalent correction-specific AO metric term.

- Stored relaxed Jacobians are model-independent only while the reference method, geometry, basis, projector basis, occupations, and numerical settings remain fixed and projector parameters are not trainable.

- DFT support must include matching nuclear RHS and grid semantics rather than only `mf.gen_response`.

- UHF and UKS must preserve the current spin-summed descriptor semantics while solving coupled alpha and beta response equations.

- Data semantics, provenance, capability checks, storage scaling, and method-specific numerical tolerances are real production requirements.

### 7.2 Required corrections

- The assessment's maintenance statements are accurate for its pinned `4f133fb` snapshot, but the current fork now declares PySCF and Torch dependencies, has a locked environment, and contains baseline tests. The remaining gap is analytic-force coverage and CI depth, not complete absence of setup and tests.

- The assessment under-describes the existing `q_P` and occupied-virtual helper functions, making the RHF implementation appear more greenfield than it is.

- The assessment's table characterizes the issue as treating degeneracy warnings as sufficient, although the issue already requires documented subspace-safe handling before exact support.

- The assessment recommends LiH as a safer low-symmetry derivative fixture. LiH is linear and has structural zero-eigenvalue degeneracy with the default high-angular-momentum projector, so it is not a generally nondegenerate fixture.

- Rejecting every exact or near-static eigenvalue gap is too blunt. Structural fixed-rank zero spaces can remain differentiable along nuclear-coordinate paths, while symmetry-breaking degeneracies can make a full ordered-eigenvalue Jacobian nonunique. The contract must test path differentiability and model sensitivity rather than use one absolute gap rule.

- The assessment omits the current ghost-center bug described below.

- The assessment's DeePKS stationarity statement needs a penalty-free and compatible-reference qualifier.

- The assessment places RHF Z-vector work before the user-visible RHF force-training vertical slice even though force training depends only on direct relaxed Jacobians.

- The assessment requests scanner cache tests before its later phase implements the scanner API.

- The assessment's eight-to-sixteen-week range is a risk estimate rather than an empirically calibrated schedule. It should be revisited after the RHF direct and RHF Z stage gates.

- The primary-source index assigns an incorrect title to arXiv `2005.00169`; the linked paper is [Ground state energy functional with Hartree-Fock efficiency and chemical accuracy](https://arxiv.org/abs/2005.00169).

## 8. Descriptor degeneracy contract

The raw descriptor consists of ordered eigenvalues of projected-density blocks. At a repeated eigenvalue, the general Hermitian-matrix-to-individual-ordered-eigenvalues map is not Frechet differentiable, and an arbitrary eigenvector derivative is insufficient for a strict force claim.

A static zero gap is not by itself a complete rejection criterion. Because `rank(O^T P O) <= n_occ`, small-electron systems can contain structural zero eigenvalues that remain identically zero while the rank is preserved.

Phase 0 must choose and test a precise contract that distinguishes structural fixed-rank zero spaces, symmetry-induced splitting, near crossings, and genuinely nondifferentiable ordered-eigenvalue behavior.

The contract must also validate model behavior. Even when a trace or thermal embedding is used, `CorrNet` retains a linear branch around the embedding in [`deepks/model/model.py`](../../deepks/model/model.py#L245-L274). Historical checkpoints, disabled preprocessing, unequal normalization, or unequal shell weights can violate the equal-sensitivity condition required inside a degenerate subspace.

The initial implementation may restrict support to fixtures and models that pass a differentiability and compatibility validator. A broader solution requires either smooth spectral invariants or a contracted subspace derivative of an explicitly symmetric energy model.

## 9. Ghost-center defect and required policy

The current explicit derivative is correct only within its ordinary real-atom convention. It is not correct for all ghost-AO geometries.

For a molecule containing a movable `X-H` ghost center, a fixed-density finite difference produces a nonzero descriptor derivative with respect to the ghost coordinate, while the current analytic `make_grad_eig_x` returns zero.

The cause is that [`deepks/scf/grad.py`](../../deepks/scf/grad.py#L43-L58) allocates derivative rows for all raw atoms but applies AO-center and projector-center derivatives only to the list filtered by `element.startswith("X")`. This removes the motion of AO basis functions located on the ghost.

The string rule is also inconsistent with PySCF aliases: `ghost-H` may appear in `mol.elements` as `GHOST-H` and therefore be misclassified as a real projector center.

The implementation must use a canonical ghost predicate based on nuclear charge or normalized PySCF atom identity, move AO basis centers for every raw atom that carries basis functions, and move projector centers only for descriptor atoms. Until this is fixed and tested, the strict method must reject unsupported ghost inputs.

## 10. Data and training requirements

The current `grad_vx` name does not need to be preserved. P0 should select canonical derivative terminology, and the implementation should update calculation APIs, stored fields, readers, and training consistently.

The public force-data schema must identify the complete relaxed descriptor Jacobian unambiguously. Explicit and response components are useful internal diagnostics, but they do not need to become separate persisted public fields unless the schema design establishes a concrete use for them.

Every response dataset must record schema version, signs, units, reference class, exchange-correlation functional, basis and ECP information, charge, spin, occupations, projector content or hash, descriptor spin semantics, geometry and atom ordering, ghost policy, software versions, SCF controls, response controls and residual, grid settings, backend identity, and differentiability diagnostics.

The current reader silently omits force data unless both force and Jacobian files exist in [`deepks/model/reader.py`](../../deepks/model/reader.py#L95-L102), and the evaluator silently skips the force term if the fields are absent in [`deepks/model/train.py`](../../deepks/model/train.py#L113-L133). A strict DeePHF force mode must validate all required data before training and fail rather than degrade to energy-only training.

Energy and force prediction should be centralized in a helper shared by training, validation, and saved-data testing. Energy and force metrics must be reported separately.

The full relaxed Jacobian scales quadratically with atom count and the current NumPy reader loads it eagerly. Schema design must support later chunked or lazy storage, but large-data optimization should not block a small-system RHF correctness milestone.

## 11. Initial support contract

The first scientific milestone should support molecular, real-orbital, integer-occupation, converged RHF references with a continuous SCF root, a well-conditioned response solve, a compatible differentiable descriptor and model, and no penalty or unsupported reference decorator.

The initial capability layer should reject ROHF, ROKS, fractional occupations, smearing, complex orbitals, periodic systems, state crossings, unsupported ghost inputs, density fitting, solvent, QM/MM, external fields, custom SCF wrappers, symmetry-constrained special occupations, and unverified scanner subclasses.

UHF should be added after the restricted response conventions are stable and after the runner stops replacing the user-specified spin in [`deepks/scf/run.py`](../../deepks/scf/run.py#L140-L155).

RKS should be added by explicit functional tiers with deterministic grid settings and step-size studies. UKS should follow both UHF spin-response stabilization and RKS grid-response stabilization.

PySCF private or semi-private Hessian helpers must be isolated behind a compatibility adapter. The initial implementation should target the locked PySCF 2.14 environment before claiming a wider supported version range.

## 12. Revised dependency sequence

The implementation should follow this dependency graph:

```text
P0 Scientific contract, architecture, and naming
 |
 v
P1 Shared descriptor core and method separation
 |
 v
P2 RHF DeePHF direct oracle
 |-------------------------|-------------------------|-------------------------|
 v                         v                         v                         v
P3A Force data/training    P3B RHF Z/inference      P4A UHF direct           P4B RKS direct
                              |                         |                         |
                              |-------------------------|--> UHF Z                |
                              |---------------------------------------------------> RKS Z
                                                        \                         /
                                                         \                       /
                                                          ---> UKS direct ----> UKS Z
                                                                    |
                                                                    v
                                                     P5 Integration and hardening
```

P1 is a separate gate so structural movement and naming changes are validated before response-theory code is added. Its contract is numerical equivalence for accepted DeePKS behavior and one-way package dependencies, not compatibility with old interfaces.

The RHF direct backend in P2 is the scientific oracle and the first response-theory stage gate.

RHF force-aware data and training do not depend on Z-vector. RHF Z-vector inference and RHF scanner validation form parallel P3 tracks after the direct oracle exists.

UHF direct and RKS direct may also start after the RHF direct interfaces stabilize. Each corresponding Z-vector backend additionally depends on the shared adjoint conventions established by the RHF Z-vector track.

UKS direct depends on stable unrestricted-spin and KS grid/XC conventions. UKS Z-vector follows its direct oracle and the accepted unrestricted and KS adjoint conventions.

Project-wide scanner, workflow, storage, compatibility, and documentation work is consolidated in P5 after the scientific tracks pass their gates.

The detailed task outline and stage gates are recorded in [`issue_93_phase_plan.md`](./issue_93_phase_plan.md).

## 13. Acceptance requirements

- Explicit descriptor derivatives must match frozen-density finite differences on ordinary atoms and the corrected ghost policy.

- First-order reference densities must match displaced-reference finite differences in a basis-aware, gauge-invariant comparison.

- Relaxed descriptors must match descriptors from independently converged displaced references over a step-size sequence.

- Direct total gradients must match finite differences of the complete perturbative `E_ref + E_corr` energy.

- Zero or constant corrections must reduce to the native reference gradient.

- Direct and Z-vector results must agree only after explicit projector, AO metric, and adjoint response contributions are all included.

- Response residual failures, occupation changes, root changes, unsupported references, incompatible models, and nondifferentiable descriptors must produce explicit failures rather than fallback results.

- Force-loss parameter gradients must match finite differences of selected model parameters.

- Force-aware training, validation, testing, and checkpoint reload must reproduce both energy and force predictions.

- Repeated scanner calls must invalidate reference, projector, descriptor, grid, and response caches correctly.

- Every supported DFT tier must document and test its grid, pruning, and grid-response semantics.

## 14. Effort assessment

An experimental RHF direct implementation remains a plausible short first milestone because PySCF supplies the forward response machinery and the repository already supplies `q_P` and explicit derivatives.

A production RHF direct backend, strict data path, force-training vertical slice, Z-vector backend, scanner, and compatibility tests remain a multi-stage effort.

Complete UHF, RKS, and UKS support is dominated by spin conventions, XC nuclear perturbations, grids, capability boundaries, and per-reference direct-versus-Z validation.

Neither the issue's three-to-five-week estimate nor the earlier assessment's eight-to-sixteen-week estimate should be treated as a commitment. Re-estimation should occur after P2 RHF direct and again after P3B RHF Z-vector.

## 15. Final assessment

Issue #93 can be completed in this fork.

The minimum defensible delivery is an RHF direct-response DeePHF method with complete relaxed descriptor derivatives, finite-difference validation, strict data semantics, and force-aware training.

The efficient RHF Z-vector backend is a subsequent optimization, not a prerequisite for the training milestone.

UHF and RKS are independent extension tracks after the RHF abstractions stabilize; UKS follows both.

The 2026-08-18 report reaches the correct central scientific conclusion but should not be executed verbatim. Its strongest contributions are the explicit-versus-relaxed diagnosis, complete `mo1` and AO metric requirements, Z-vector metric correction, DFT grid warning, data provenance requirements, and production-scope caution. Its implementation sequence, degeneracy characterization, current-repository context, and unreported ghost and penalty boundaries require the revisions in this update.
