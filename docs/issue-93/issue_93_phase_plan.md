# Issue #93 Implementation Phase Plan

**Objective:** Complete [deepmodeling/deepks-kit Issue #93](https://github.com/deepmodeling/deepks-kit/issues/93) in the `light-cyan/deepks-kit` fork by delivering validated analytic DeePHF nuclear forces and force-aware training for a declared support domain.

**Planning level:** This roadmap defines phase objectives, dependencies, major work, and exit gates. Each phase receives a separate detailed task document when implementation starts.

**Current position:** P0, P1, P2, P3A, P3B, P4A, P4B, and P4C are complete; the finite-grid closed-shell pure-LDA RKS scalar-adjoint backend is accepted against the P4B direct oracle and deterministic finite differences.

**Technical basis:** [2026-08-19 assessment](./deepks_issue_93_assessment.0819.md)

## 1. Planning principles

- Self-consistent DeePKS and perturbative DeePHF belong to separate method packages and must not import one another.
- A shared descriptor layer contains only method-independent projection, descriptor, and local derivative operations required by both methods.
- The refactor preserves verified scientific behavior, not old import paths, field names, aliases, or configuration forms.
- Canonical names follow the mathematical object and derivative meaning; package context may shorten a name only when its meaning remains unique.
- Direct response is the correctness oracle; each Z-vector backend follows its matching validated direct backend.
- Force data carries machine-checkable derivative semantics. Existing fixed-density data cannot supply the missing response and must be regenerated for exact DeePHF training.
- Unsupported references, response failures, incompatible data, and nondifferentiable cases fail explicitly.

## 2. Dependency map

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
P3A Force data/training    P3B RHF Z/inference      P4A UHF direct [done]    P4B RKS direct [done]
                              |                         |                         |
                              |-------------------------|--> UHF Z                |
                              |---------------------------------------------------> RKS Z [P4C]
                                                        \                         /
                                                         \                       /
                                                          ---> UKS direct ----> UKS Z
                                                                    |
                                                                    v
                                                     P5 Integration and hardening
```

P3A, P3B, UHF direct, and RKS direct may start after the P2 oracle and shared interfaces are stable.

Each UHF or RKS Z-vector backend depends on its direct oracle and the shared adjoint conventions established in P3B. UKS follows the accepted unrestricted-spin and KS grid/XC conventions.

P5 begins after the accepted P3 and P4 delivery tracks pass their gates.

## 3. P0 — Scientific contract, architecture, and naming

**Goal:** Define what the two methods mean and where each responsibility belongs before moving or extending response-sensitive code.

**Major work:**

- Define the supported scientific scope, mathematical quantities, derivative semantics, tensor conventions, and failure boundaries.
- Specify the shared descriptor boundary, the independent DeePKS and DeePHF method boundaries, and the permitted dependency direction.
- Establish one canonical vocabulary for code, APIs, data, and training without designing a legacy compatibility layer.
- Characterize reusable primitives and protect current valid DeePKS numerical behavior with deterministic finite-difference and regression tests.

**Exit gate:** The scientific contract, package boundaries, dependency rules, naming vocabulary, and numerical baseline are accepted.

## 4. P1 — Shared descriptor core and method separation

**Goal:** Complete the structural refactor before adding DeePHF response theory.

**Major work:**

- Extract method-independent projector, descriptor, and local derivative capabilities into a shared descriptor package.
- Move self-consistent correction, SCF, variational-gradient, and method-specific penalty behavior into the DeePKS package.
- Establish an independent DeePHF package that composes around a native converged reference and consumes the shared descriptor interface.
- Move method-neutral field and data responsibilities out of method internals and apply the P0 naming vocabulary throughout the refactored path.

**Exit gate:** The shared layer imports neither method package, DeePKS and DeePHF do not import one another, descriptor logic is not duplicated, and DeePKS numerical regressions pass after the old mixed layout is removed.

## 5. P2 — RHF DeePHF direct oracle

**Goal:** Deliver the first complete analytic DeePHF gradient for a converged RHF reference inside the independent DeePHF package.

**Major work:**

- Implement the complete RHF relaxed descriptor response and total DeePHF gradient through a bounded PySCF response interface.
- Keep explicit descriptor motion, reference-density response, and total relaxed response distinguishable in internal validation.
- Validate density response, descriptor response, and total gradients against independently converged displaced-reference finite differences.

**Exit gate:** RHF direct gradients pass the accepted scientific checks and become the oracle for training, Z-vector work, and reference expansion.

## 6. P3 — RHF end-to-end delivery

P3 contains two parallel tracks after P2.

### P3A — Force data and training

**Status:** Complete. The strict v1 RHF force-data contract, direct-response producer, relaxed-Jacobian reader and contraction, force-aware training and validation, saved-data testing, compatible checkpoint reload, deterministic acceptance tests, and runnable example are implemented and documented in [P3A RHF relaxed-force data and training](./p3a_rhf_force_training.md).

- Define the force-data schema around the complete relaxed descriptor Jacobian and required provenance.
- Update data generation, readers, training, validation, saved-data testing, and checkpoint workflows to use the canonical semantics.
- Demonstrate a compact RHF energy-and-force training workflow and reject incomplete or ambiguous data.

### P3B — RHF Z-vector inference

**Status:** Implemented and target-accepted. The strict RHF scalar-adjoint backend, reference-neutral adjoint protocol, fresh-reference direct/Z-vector scanner, deterministic algebra and finite-difference tests, and failure-state checks are implemented and documented in [P3B RHF DeePHF Z-vector inference](./p3b_rhf_zvector_inference.md).

- Implement RHF adjoint response and validate it against the direct oracle and total-energy finite differences.
- Establish shared adjoint interfaces for later reference classes.
- Implement and validate RHF scanner behavior across geometry changes.

**Exit gate:** RHF force-aware training and efficient RHF molecular inference both work end to end with direct, Z-vector, and finite-difference agreement.

## 7. P4 — Reference-method expansion

**Goal:** Extend the accepted DeePHF architecture to UHF, RKS, and UKS in a controlled order.

**Major work:**

- The accepted UHF and closed-shell pure-LDA RKS direct oracles serve as correctness references under their separate spin, functional, and finite-grid conventions.
- Add UHF and RKS Z-vector backends after their direct oracles and the P3B adjoint foundation are accepted.
- Add UKS direct after the UHF and RKS conventions are stable, then add UKS Z-vector.
- Maintain and test an explicit capability matrix for every advertised reference and XC tier.

**Exit gate:** Every advertised UHF, RKS, and UKS tier passes total-energy finite differences, direct-versus-Z agreement, and capability checks.

### P4A — UHF direct oracle

**Status:** Implemented and target-accepted. The exact native UHF validator, complete coupled alpha/beta UC-PHF density response, additive spin/metric/occupied-virtual descriptor and gradient partitions, strict response audits, deterministic finite-difference oracle, and package-isolation guards are implemented and documented in [P4A UHF DeePHF direct oracle](./p4a_uhf_direct_oracle.md).

- Preserve the canonical spin-summed descriptor while exposing additive alpha and beta density, descriptor-response, and correction-gradient partitions.
- Retain the complete coupled occupied-virtual and AO-metric response as an auditable direct oracle.
- Keep the RHF force-data, scalar-adjoint, and scanner implementations isolated from the UHF direct object graph.

### P4B — RKS direct oracle

**Status:** Implemented and target-accepted. The exact native closed-shell RKS validator, characterized LibXC 7.0.0 pure-LDA tier, deterministic unpruned atom-centered grid contract, complete Coulomb plus `f_xc` CPKS response, hardened grid-response and `w1` audits, bounded trusted-response reuse, AO-metric and grid-motion partitions, audited native grid-response gradient, finite-difference and cross-molecule oracles, and package-isolation guards are implemented and documented in [P4B RKS DeePHF direct oracle](./p4b_rks_direct_oracle.md).

- The oracle binds the exact finite-grid energy to normalized `LDA_X + LDA_C_VWN` semantics under LibXC `7.0.0`, `NumInt.cutoff=1e-13`, grid cutoff `1e-15`, and byte-reproducible `(20, 50)` atom-grid provenance with the canonical PySCF 2.14 `BRAGG_RADII` content fingerprint.
- The exact response-generator identity, host-atom block boundaries, cached energy-grid weights, full translational `w1`, and independent `h=1e-5 Bohr` grid-weight finite differences are enforced before CPKS, while provenance records the qualified response-generator identity and weight-derivative fingerprint.
- Runtime responses retain fixed-grid XC AO motion, grid-coordinate response, grid-weight response, AO-metric response, occupied-virtual response, and complete relaxed descriptor and gradient values as auditable partitions, and each method can reuse up to eight responses that it produced and continues to revalidate.
- Strict `H2` and `LiH` cross-molecule smoke checks exercise the characterized native functional and deterministic grid domain.
- Architecture guards keep the RHF force-data, scalar-adjoint, and scanner implementations and the UHF direct implementation isolated from the RKS direct object graph.

### P4C — RKS scalar-adjoint inference

**Status:** Implemented and target-accepted. The correction-specific finite-grid RKS transpose solve, complete fixed-grid and moving-grid nuclear contractions, fail-closed inference driver, independent dense-grid oracle, and strict fault matrix are documented in [P4C RKS DeePHF Z-vector inference](./p4c_rks_zvector_inference.md).

- Reuse the P3B reference-neutral transpose convention and the P4B physical Coulomb plus dense-LDA-`f_xc` operator without constructing a coordinate-wise density response.
- Retain objective-metric, fixed-grid, grid-coordinate, grid-weight, adjoint-metric, occupied-virtual, response, correction, native-reference, and total gradient partitions.
- Preserve the P4B direct oracle as the default backend and as the model-independent coordinate-wise response facility.

## 8. P5 — Integration and hardening

**Goal:** Turn the validated scientific backends into a maintainable project-wide feature.

**Major work:**

- Integrate the separated methods into public CLI, workflow, data-generation, scanner, and geometry-optimization paths.
- Extend lifecycle and compatibility testing across all accepted backends.
- Address storage and performance limits based on measurements from supported use cases.
- Finalize examples, capability documentation, and release verification.

**Exit gate:** All advertised modes pass the complete regression and build workflow through their public entry points, with documented support and measured operating limits.

## 9. Execution notes

- Create a phase-specific task document when a phase starts; keep implementation-level decisions out of this roadmap.
- Re-estimate remaining work after P2 and after P3.
- Parallel tracks must preserve the conventions established by their dependency gates.
