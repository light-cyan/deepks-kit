# P1 Shared Descriptor and Method-Separation Task

## 1. Objective

Complete the structural refactor defined by the [P0 scientific contract](./p0_scientific_contract.md) before adding reference-response theory.

The delivered tree has one shared `deepks.descriptor` implementation, an independent self-consistent `deepks.deepks` method, an independent perturbative `deepks.deephf` composition, and a method-neutral `deepks.data` layer.

The refactor preserves validated DeePKS numerical behavior and intentionally exposes only the canonical P0 interfaces.

## 2. Tracked move map

The branch uses the following tracked moves as the starting point; each rename or move is performed with `git mv` so Git retains file identity.

| Source | Tracked destination | Retained responsibility |
|---|---|---|
| `deepks/scf/__init__.py` | `deepks/deepks/__init__.py` | Public self-consistent DeePKS package entry. |
| `deepks/scf/__main__.py` | `deepks/deepks/__main__.py` | DeePKS module entry point. |
| `deepks/scf/scf.py` | `deepks/deepks/method.py` | Self-consistent DeePKS method classes after shared descriptor extraction. |
| `deepks/scf/grad.py` | `deepks/deepks/gradient.py` | DeePKS variational gradient classes after shared local-derivative extraction. |
| `deepks/scf/penalty.py` | `deepks/deepks/penalty.py` | DeePKS-only penalty behavior. |
| `deepks/scf/addons.py` | `deepks/deepks/addons.py` | DeePKS-only potential and optimization helpers. |
| `deepks/scf/run.py` | `deepks/deepks/run.py` | DeePKS runner after neutral I/O extraction. |
| `deepks/scf/fields.py` | `deepks/data/fields.py` | Neutral field primitives after method-specific producers are separated. |
| `deepks/scf/stats.py` | `deepks/data/stats.py` | Method-neutral statistics. |

No moved file may retain an import from `deepks.scf`, and the refactor does not recreate `deepks/scf` as a compatibility package.

## 3. Target layout

The P1 target has the following responsibility layout:

```text
deepks/
  descriptor/
    __init__.py
    core.py
    derivatives.py
    projection.py
    orbitals.py
  deepks/
    __init__.py
    __main__.py
    method.py
    gradient.py
    penalty.py
    addons.py
    run.py
  deephf/
    __init__.py
    capabilities.py
    method.py
  data/
    __init__.py
    fields.py
    io.py
    stats.py
  model/
    evaluate.py
```

Files already present outside this layout remain in their current package when their responsibility is unchanged.

## 4. Shared descriptor extraction

Move or extract the following active facilities into `deepks.descriptor` without duplicating their implementations:

- `core.py` owns evaluation of `D = O^T P O`, shell-wise Hermitian eigenvalues, deterministic shell concatenation, and the canonical `q` axis layout.

- `derivatives.py` owns batched Jacobian support, `partial q / partial P`, `dD/dR` at fixed `P`, and `dq_dR_explicit` at fixed `P`.

- `projection.py` owns projector-molecule construction, projection-basis normalization, shell layout, `O` and derivative-integral construction, integral-cache reset, and the explicit raw-atom-to-descriptor-atom mapping.

- `orbitals.py` owns the method-independent AO-matrix to occupied-virtual coordinate transformation used by response-sensitive helpers.

- `__init__.py` exports a small canonical surface for `P`, `O`, `D`, `q`, `dq_dR_explicit`, and `partial q / partial P` operations without exporting method classes.

Replace projection behavior embedded in the DeePKS mixin with composition around a shared descriptor object that owns the molecule, projector basis, shell layout, atom mapping, and integral cache.

Use `mol.atom_charge(A) == 0` for ghost identity, move AO centers for all raw atoms with AO functions, and move projector centers only for the explicitly mapped descriptor atoms.

Move correction-model evaluation from descriptor or method code into `deepks.model.evaluate`; this module evaluates `e_corr` and its derivatives from `q` but does not own projection mathematics or method state.

## 5. Self-consistent DeePKS separation

Keep corrected Fock construction, corrected SCF classes, stationary total-energy evaluation, variational gradients, penalty behavior, and method-specific scanner behavior inside `deepks.deepks`.

Use `DeePKS` for the explicit public method factory and `RDeePKS` and `UDeePKS` for restricted and unrestricted concrete method classes.

Remove generic names and aliases whose meaning depends on the former mixed package, including `DSCF`, `UDSCF`, `DeepSCF`, `Grad`, and `RGrad`.

Make `nuc_grad_method()` call the DeePKS gradient factory in one dependency direction; the gradient module must not import concrete DeePKS classes for module-bottom method injection.

Keep all penalty imports and behavior inside `deepks.deepks`, and reject strict force fields for a state whose reported energy lacks matching penalty derivatives.

Expose DeePKS-specific field values through stable methods on the DeePKS method and gradient objects so `deepks.data.fields` can remain independent of method implementation imports.

## 6. Independent perturbative DeePHF composition

Implement `deepks.deephf.DeePHF` as composition around an already converged native reference, a correction model, and the shared descriptor object.

P1 DeePHF exposes `P`, `D`, `q`, `e_base`, `e_corr`, and `e_tot` while leaving the reference Fock, orbitals, occupations, and convergence state unchanged.

`deepks.deephf.capabilities` implements the strict native-RHF and model checks that are meaningful before response execution, including convergence, real orbitals, integer occupations, undecorated reference identity, projector compatibility, ghost mapping, and descriptor differentiability.

P1 does not expose a DeePHF analytic force until a validated relaxed response backend supplies `dq_dR_relaxed`.

The DeePHF package imports shared descriptor and model interfaces and never imports DeePKS method, gradient, penalty, addon, field, or runner modules.

## 7. Neutral data separation

Keep `Field`, canonical field selection, unit conversion, molecular input, generic array serialization, and statistics in `deepks.data`.

Make unknown or duplicate field names an explicit error rather than silently omitting them.

Use the canonical fields `ao_density`, `projected_density`, `descriptor`, `dD_dR_explicit`, `dq_dR_explicit`, `e_base`, `e_corr`, `e_corr_target`, `e_tot`, `f_reference_variational`, `f_corr_explicit`, `f_corr_explicit_target`, and `f_tot` where each corresponding quantity is valid.

Remove field aliases such as `dm_eig`, `eig`, `grad_vx`, and `gvx`; `grad_vx` has only fixed-density semantics and must not survive as an ambiguous force-data field.

Move system reading, data aggregation, and array writing from the method runner to `deepks.data.io`; the DeePKS runner selects a method and supplies public results to neutral I/O.

Do not import `deepks.deepks.addons` from `deepks.data`; optional DeePKS calculations are exposed through stable methods on the method object.

## 8. Consumer migration

Update production consumers in `deepks/__init__.py`, `deepks/main.py`, `deepks/iterate`, `deepks/tools`, and `deepks/model` to use canonical package imports and field names.

Update `deepks/iterate/template.py` so generated commands select `deepks.deepks.run` rather than the mixed SCF module.

Update readers, training, and saved-data tests to use canonical field names and to distinguish `dq_dR_explicit` from force-capable `dq_dR_relaxed`.

Update active examples and configurations to use canonical field names without alias fallback.

Update baseline tests to import descriptor functions from `deepks.descriptor`, DeePKS functions from `deepks.deepks`, and neutral facilities from `deepks.data`.

## 9. Required tests

Add an import-boundary test that inspects the package dependency graph and fails if `deepks.descriptor` imports a method package, either method imports the other, or `deepks.data` imports method internals.

Add a public-surface test that verifies canonical packages and names import and that `deepks.scf`, mixed-package aliases, and ambiguous field aliases are unavailable.

Retain deterministic descriptor value, AO-density Jacobian, explicit nuclear Jacobian, nonzero DeePKS energy, total-gradient, scanner, field, CLI, example, model-driven SCF, checkpoint, and pipeline regressions.

Add fixed-`P` central finite-difference tests for ordinary centers, `X-H`, and `ghost-H`, including the raw-atom and descriptor-atom axis mapping and translational sum rule.

Add a DeePHF composition test showing that `e_tot = e_base + e_corr`, that the native converged reference state is unchanged, and that no analytic-gradient API is exposed by the P1 energy composition.

Add strict capability tests for accepted native RHF input and explicit rejection of decorated, unconverged, fractional-occupation, complex-orbital, penalty-bearing, and differentiability-incompatible inputs.

## 10. Execution sequence

1. Record `git status --short --branch`, run the protected baseline suite, and preserve any unrelated user changes.

2. Complete the tracked moves in the move map with `git mv`, then update imports immediately so each intermediate package remains inspectable.

3. Extract shared descriptor and model-evaluation facilities, replace inheritance-owned projection state with descriptor composition, and add numerical equivalence tests before changing method consumers.

4. Separate neutral data facilities and DeePKS-specific field producers, then apply canonical field names through readers, training, tests, and active configurations.

5. Establish the independent DeePHF energy composition and strict P1 capability validation without adding response or force claims.

6. Confirm with `rg` and import tests that active consumers use the separated packages and canonical fields.

7. Run the objective checks, the complete suite, import-boundary checks, and repository residue checks before presenting the diff for review.

## 11. Exit verification

P1 exits only when every condition below is satisfied:

- `deepks.descriptor` contains the only active implementation of shared projection, descriptor, and local derivative mathematics.

- `deepks.deepks` and `deepks.deephf` are independent sibling packages and neither imports the other.

- `deepks.data` is method-neutral and contains no import of method internals.

- The `deepks.scf` directory, imports, commands, configuration keys, and compatibility aliases are absent from active code, tests, and examples.

- `dq_dR_explicit` is the only name for the fixed-`P` descriptor derivative and is never consumed as force-capable relaxed data.

- Ordinary and ghost-center explicit descriptor derivatives pass their deterministic finite-difference and translation checks.

- Accepted penalty-free DeePKS energies, descriptors, and total gradients match the protected numerical baselines within their existing explicit tolerances.

- P1 DeePHF energy composition passes without mutating the native reference and exposes no analytic force interface.

- All relevant tests and the complete repository suite pass.

Run the following verification commands from the repository root:

```bash
git status --short --branch
rg -n "deepks\.scf|from deepks import scf|python -m deepks\.scf|grad_vx|\bgvx\b|dm_eig" deepks examples -g '!examples/legacy/**'
uv run pytest tests/baseline
uv run pytest
uv run python -c "import deepks.descriptor, deepks.deepks, deepks.deephf, deepks.data"
uv run python -c "import importlib.util; assert importlib.util.find_spec('deepks.scf') is None"
```

The residue search must return no active compatibility use; a canonical-to-obsolete-name assertion inside a rejection test may remain when it is clearly scoped to that test.

Review `git diff --stat`, `git diff --summary`, and `git diff` after verification so tracked renames, extracted code, numerical tests, and documentation are visible before any commit is requested.
