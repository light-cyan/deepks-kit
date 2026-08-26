# Scientific Diagnostic Findings

## Scope

This report records the scientific-validation protocol repair executed from base revision `802a40c15bc5be7075028d66c9e0011a98dfc0ab` in the current working tree. The production final-density Fock canonicalization and the default `conv_tol=1e-12` remain unchanged.

## Effective finite-difference protocol

Selected-coordinate workloads use the default predeclared steps `1e-3` and `3e-4 Bohr`. L1/def2-TZVP RHF uses family-specific steps `8e-4` and `3e-4 Bohr`. L3/def2-SVP UHF uses `conv_tol_grad=1e-7` with steps `2.5e-3` and `2e-3 Bohr`. L3/def2-SVP UKS uses `conv_tol_grad=1e-8` with steps `3e-3` and `2e-3 Bohr`.

Every declared step is evaluated over the complete configured Cartesian component set and five deterministic full-coordinate directions. Each step is independently checked against the unchanged `1e-5 Ha/Bohr` force threshold and `1e-5 Bohr^-1` relaxed-descriptor threshold.

## Candidate validation

The first complete rerun of the eight previously failing cases completed without process, integrity, resource, reference-convergence, or state-continuity failures. Seven cases passed. L1/def2-TZVP RHF failed only at `1e-3 Bohr`, where the maximum relaxed-descriptor error was `1.211572078e-5 Bohr^-1` at atom 0, x, descriptor index `[0,2]`; its `3e-4 Bohr` error was `1.090182853e-6 Bohr^-1`.

The dedicated L1/def2-TZVP RHF rerun evaluated all twelve configured coordinates and five directions at both replacement candidates. At `8e-4 Bohr`, the maximum force, relaxed-descriptor, and directional errors were `1.526269060e-7 Ha/Bohr`, `7.753415458e-6 Bohr^-1`, and `7.881741710e-8 Ha/Bohr`. At `3e-4 Bohr`, they were `2.421921518e-8 Ha/Bohr`, `1.090182853e-6 Bohr^-1`, and `9.776347936e-9 Ha/Bohr`. Both steps passed every acceptance check.

## Complete scientific matrix

All fourteen configured RHF, RKS, UHF, and UKS scientific cases passed in `runs/protocol_repair_full14_20260826`. All fourteen process, scientific, integrity, and resource outcomes passed, with no timeout or invalid reference state.

Across the complete matrix, the maximum complete-energy component error was `2.815362454e-6 Ha/Bohr`, the maximum directional error was `4.010525408e-7 Ha/Bohr`, the maximum relaxed-descriptor error was `9.230047660e-6 Bohr^-1`, and the maximum direct-versus-Z-vector error was `2.273e-12 Ha/Bohr`. Every predeclared step passed independently.

Zero-correction, complete energy-force finite differences, five-direction derivatives, relaxed descriptors, direct-versus-Z-vector, compact-versus-detailed, repeated-input, checkpoint-reload, canonical residual, electron-count, state-continuity, and result-integrity checks passed in every case.

## Source provenance

The complete matrix executed directly from `/home/mwding/WorkSpace/Projects/deepks-kit` with `source_mode=current-worktree`. The recorded source was dirty, its base revision was `802a40c15bc5be7075028d66c9e0011a98dfc0ab`, its tracked diff SHA-256 was `8af57b2788928c8d4ca7756441d1c36d14cccd7a5bd086623e3a9245b6e21efb`, its configuration SHA-256 was `1cf16c2be363bd49b3692a8f986b2d46e3ff52f4de90a49c02d434e51b196002`, and its validation-input SHA-256 was `8dd9ad6bf10b4aca7c10d2d9c1f4481529f5b6e1a187bd3f710a7de621a32484`.

All fourteen child results record the same source snapshot, and the aggregate source-provenance, validation-hash, affinity, and thread-control checks passed.

## Regression verification

The deterministic campaign-protocol tests pass with `3 passed`. The related analytic-force, scalar-adjoint, and scientific-protocol objectives pass with `622 passed`. The complete suite passes with `897 passed`.

## Artifacts

The initial eight-case evidence is under `runs/protocol_repair_failed8_20260825`, the dedicated L1/def2-TZVP RHF confirmation is under `runs/protocol_repair_l1_tzvp_rhf_20260825`, and the accepted complete matrix is under `runs/protocol_repair_full14_20260826`. Each run contains a manifest, machine-readable aggregate, Markdown summary, and per-case result, controller, stdout, stderr, and resource files.
