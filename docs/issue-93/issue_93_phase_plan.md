# Issue #93 Implementation Phase Plan

**Objective:** Complete [deepmodeling/deepks-kit Issue #93](https://github.com/deepmodeling/deepks-kit/issues/93) in the `light-cyan/deepks-kit` fork by delivering validated analytic DeePHF nuclear forces and force-aware training for a declared support domain.

**Planning level:** This roadmap defines phase objectives, dependencies, major work, and exit gates. Each phase receives a separate detailed task document when implementation starts.

**Current position:** P0, P1, P2, and P3A are complete; strict RHF DeePHF relaxed-force data, force-aware training, validation, saved-data testing, and checkpoint reload are accepted, with the validated P2 direct oracle providing their scientific response semantics.

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

- Implement RHF adjoint response and validate it against the direct oracle and total-energy finite differences.
- Establish shared adjoint interfaces for later reference classes.
- Implement and validate RHF scanner behavior across geometry changes.

**Exit gate:** RHF force-aware training and efficient RHF molecular inference both work end to end with direct, Z-vector, and finite-difference agreement.

## 7. P4 — Reference-method expansion

**Goal:** Extend the accepted DeePHF architecture to UHF, RKS, and UKS in a controlled order.

**Major work:**

- Develop UHF direct and RKS direct as parallel tracks after P2, validating spin and grid/XC conventions independently.
- Add UHF and RKS Z-vector backends after their direct oracles and the P3B adjoint foundation are accepted.
- Add UKS direct after the UHF and RKS conventions are stable, then add UKS Z-vector.
- Maintain and test an explicit capability matrix for every advertised reference and XC tier.

**Exit gate:** Every advertised UHF, RKS, and UKS tier passes total-energy finite differences, direct-versus-Z agreement, and capability checks.

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
