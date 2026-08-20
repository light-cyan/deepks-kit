# P4E UKS DeePHF Direct and Z-Vector Inference

## 1. Status and objective

P4E implements strict direct-response and scalar-adjoint analytic DeePHF gradients for native converged open-shell pure-LDA UKS references under the characterized PySCF 2.14, LibXC 7.0.0, and deterministic finite-grid contract.

The canonical descriptor consumes the spin-summed AO density `P_alpha + P_beta`. The direct backend retains complete alpha, beta, total, AO-metric, occupied-virtual, fixed-grid XC AO-motion, grid-coordinate, grid-weight, native-reference, correction, and total derivative partitions. The scalar-adjoint backend obtains the same scalar total gradient with one coupled alpha/beta transpose solve.

## 2. Public API and ownership

`UKSDeePHF` composes the correction around an exact native `pyscf.dft.uks.UKS` reference. `UKSDeePHFGradients` is the default coordinate-wise direct oracle, and `UKSDeePHFZVectorGradients` is selected with `backend="zvector"`.

`UKSResponseAdapter`, `UKSAdjointAdapter`, `UKSResponse`, `UKSAdjoint`, their diagnostics, and `UKSNativeGradient` are exported from `deepks.deephf`. PySCF UKS Hessian, gradient, NumInt, LibXC, and molecular-state integration remains in `deepks.deephf.pyscf_uks`.

The direct response is model-independent and provides `first_order_spin_density`, spin-summed `first_order_density`, `dq_dR_response`, and `dq_dR_relaxed`. The scalar adjoint is model-specific inference state and performs no coordinate-wise density-response solve.

## 3. Strict reference and finite-grid contract

The accepted reference has exact native UKS and Mole types, a converged open-shell Aufbau state, complete real `numpy.float64` alpha and beta canonical orbitals, zero-or-one occupations matching `mol.nelec` and `mol.spin`, occupied and virtual spaces in both channels, finite symmetric spin densities, correct AO-metric electron counts and idempotency, a direct dense-grid effective potential, canonical residuals, and reconstructed total energy.

The accepted XC functional is normalized `LDA_X + LDA_C_VWN` with the characterized native LibXC 7.0.0 backend, exact `NumInt.cutoff=1e-13`, no hybrid, range separation, NLC, or custom functional, and a spin-polarized canonical `exc`, `vxc`, and `fxc` signature.

The finite grid uses the same accepted unpruned per-element `(20, 50)` atom grid, alignment, cutoff, radial method, radii, Becke partition, host-block, cached-weight, and independently audited weight-derivative contract as strict RKS.

## 4. Coupled direct response

For spin `sigma`, a trial occupied-virtual amplitude produces `delta P_sigma = C_sigma,v X_sigma C_sigma,o.T + transpose`. The physical UKS action contains `J[delta P_alpha + delta P_beta]` and the full spin-polarized LDA kernel `sum_tau f_xc[sigma,tau] delta rho_tau`, so the alpha and beta channels are solved as one coupled operator.

The nuclear right-hand side contains the overlap metric term, core and Coulomb derivative, fixed-grid spin XC AO motion, atom-centered grid-coordinate motion, Becke grid-weight motion, and the induced potential of both spin metric densities. Every returned response is independently reaudited without another CPHF solve.

## 5. Coupled scalar adjoint

The spin-summed descriptor gives one symmetric AO objective `W` acting in both spin channels. The bilateral occupied-virtual objective is `b_sigma[a,i] = W_sigma[a,i] + W_sigma[i,a]`, and one reference-neutral solve evaluates `A.T [z_alpha,z_beta] = [b_alpha,b_beta]`.

The correction gradient retains objective-metric, fixed-grid adjoint, grid-coordinate adjoint, grid-weight adjoint, induced-potential metric, occupied-virtual, response, explicit, and total partitions. The fixed-grid, grid-coordinate, and grid-weight adjoint terms sum exactly to the complete nuclear adjoint contraction, while direct and scalar-adjoint total gradients agree.

## 6. Numerical acceptance

The deterministic distorted `NH2/STO-3G` doublet fixture uses the accepted pure-LDA finite grid and a nontrivial double-precision nonlinear correction model. Three central-displacement step sizes validate alpha, beta, and total first-order AO densities, relaxed descriptors, and complete `e_base + e_corr` gradients against independently converged fresh UKS references.

The coupled operator is symmetric, positive, finite, and well conditioned on the accepted fixture. Grid-coordinate and grid-weight contributions are separately nonzero, direct and scalar-adjoint gradients agree, the scalar adjoint reports one solve, and zero or constant corrections reproduce the complete native UKS grid-response gradient.

## 7. Verification

```bash
uv run pytest -q tests/uks_analytic_forces
uv run pytest -q tests/uks_zvector_inference
uv run pytest -q tests/analytic_forces/test_package_architecture.py
```

Final Python 3.11 acceptance contains 25 UKS direct and scalar-adjoint tests, including three-step fresh-reference density, descriptor, and total-energy finite differences, public-result audits, backend independence, scalar-control faults, operator conditioning, reference convergence, and fail-closed driver state.
