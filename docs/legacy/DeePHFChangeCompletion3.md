# DeePHF Change Completion 3

## Result

The corrective DeePHF refactor defined by `DeePHFChangeRequirements3.md` is complete. The current implementation provides one canonical CorrNet execution-state contract, tracked built-in setters, calculation-scoped cache validation, a trusted controlled workflow, explicit PySCF compatibility facades, consolidated algorithm modules, and focused state-safety, performance, topology, and scientific regression tests.

## Model Execution State

`deepks/model/model.py` owns `force_model_structure_evidence`, `model_execution_state_evidence`, and `model_execution_state_fingerprint`. DeePHF cache validation, scanner validation, force-model validation, checkpoint validation, model evaluation, and force-aware training consume this contract directly or through a compatibility delegate.

The contract records exact CorrNet and module identities, implementation and dispatch identities, compilation and hook state, training modes, input and projector metadata, element and embedder configuration, DenseNet activation and residual configuration, layer topology, and parameter and buffer identity, storage, metadata, mutation version, and numerical values for complete fingerprints.

Generic `torch.nn.Module` energy models use a conservative complete model fingerprint at public reuse boundaries and discard cached model outputs after validation. Exact CorrNet models use cheap execution-state evidence before conservative reference and descriptor hashing.

CorrNet normalization, prefitting, and energy-constant setters update registered parameters with tracked `copy_` operations under `torch.no_grad()`. Their tests cover mutation versions, object identity, trainability, dtype, device, in-transaction rejection, and independent-calculation rebinding.

## Evaluation and Cache Budgets

Method construction performs one complete scientific-state binding fingerprint, including one complete model fingerprint, outside the calculation operation report.

The deterministic nonzero RHF energy-gradient-descriptor sequence reports the following calculation-scoped operations:

| Workflow | Complete scientific-state fingerprints | Complete model fingerprints | Cheap evidence validations | Conservative cache-state fingerprints | Descriptor evaluations | Model forwards | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Controlled package workflow | 2 | 2 | 0 | 0 | 1 | 1 | 10 |
| Public user-interruptible workflow | 2 | 2 | 2 | 2 | 1 | 1 | 10 |

Fresh public contexts begin calculation without a cache-state fingerprint. Later public boundaries validate cheap model evidence and conservatively hash reference and descriptor buffers before consuming calculation-scoped cached values. Controlled `evaluate_molecule` and detailed force-data workflows share one accepted state token and evaluation context across uninterrupted internal operations.

## Package Topology

The current `deepks/deephf/` inventory uses the counting definitions in `DeePHFChangeRequirements3.md`.

| Metric | Result |
| --- | ---: |
| Python files | 41 |
| Top-level Python files | 32 |
| Audit Python files | 9 |
| Total physical lines | 17,609 |
| Maximum module size | 908 lines in `pyscf_dft_provenance.py` |
| Maximum function size | 198 lines in `pyscf_rks_adjoint.py:_solve` |

Direct-gradient family drivers share `gradient.py`, scalar-adjoint family drivers share `zvector.py`, and the two algorithms remain physically independent. Unrestricted method and reference implementations, scanner construction, RKS reference behavior, and audit responsibilities use cohesive shared owners. Dense audit implementations are loaded lazily from compact production paths.

The `pyscf_rhf.py`, `pyscf_uhf.py`, `pyscf_rks.py`, and `pyscf_uks.py` compatibility modules define explicit `__all__` surfaces. Facade topology tests enforce the deliberate exports, implementation-free modules, and clean star-import namespaces.

Restricted density-from-MO-response contraction is owned by `RestrictedResponseAlgebra`, and gradient scanner construction is owned by `GradientDriver`. Family method classes inherit shared constructor behavior when their argument contract is unchanged.

## Regression Coverage

The dedicated state-safety objective covers activation and residual configuration, residual scaling, embedder and element configuration, hooks, compilation dispatch, training mode, model and layer replacement, parameter and buffer replacement, storage mutation, conventional `no_grad` mutation, input and projector metadata, built-in setters, generic model reuse, and reference-array mutation. Mutation failures clear all transaction publishers before returning a cached dependent result.

Architecture tests enforce aggregate file counts, module and function size limits, direct-versus-adjoint separation, generic adjoint dependency independence, lazy audit imports, explicit facade exports, tracked CorrNet mutation, and canonical ownership boundaries.

## Verification

The locked Python 3.11 environment synchronized successfully with `uv sync --locked --python 3.11`.

The complete suite passed with 885 tests using `uv run pytest -q`.

Source distribution and wheel construction passed with `uv build`.

`git diff --check` passed.
