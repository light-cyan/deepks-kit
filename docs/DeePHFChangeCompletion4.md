# DeePHF Change Completion 4

## Result

The focused correction defined by `DeePHFChangeRequirements4.md` is complete. Public user-interruptible cache reuse now validates current model values, the canonical CorrNet graph contract binds project-owned thermal-embedding helpers, and the existing 41-file DeePHF layout has one mechanically clean source structure per consolidated owner.

## Public Cache Safety

Every public boundary that can consume calculation-scoped cached values first compares cheap scientific-state evidence. A matching cheap model state is followed by the canonical complete model fingerprint, which hashes current parameter and buffer values before the cached result can return.

Tracked tensor mutations, model graph changes, and trusted helper replacements fail through cheap evidence. The rejecting boundary performs the complete value hash only after cheap evidence matches, so `.data` and writable shared NumPy alias mutations are detected without adding fallback work after a known cheap-evidence mismatch.

Failed transactions atomically clear method energy fields and registered gradient results. A later independent calculation accepts a model-only change at calculation entry and evaluates the updated parameter or buffer values.

Controlled `evaluate_molecule` and force-data paths remain uninterrupted. Their accepted transaction token supports one descriptor evaluation, one model forward, and zero intermediate public model-value or conservative cache-state fingerprints.

## Canonical Helper Contract

`deepks/model/model.py` owns the explicit `_TRUSTED_CORRNET_HELPERS` tuple for `pad_masked`, `masked_softmax`, and `unpad_masked`. ThermalEmbedding graph evidence records the current callable and code identities, and force validation compares them with the trusted implementations.

The canonical evidence also binds `ThermalEmbedding.update_running_stats`. Complete fingerprints change for a replaced helper and return to their original value when the trusted implementation is restored. Models with no embedder and TraceEmbedding preserve their established numerical paths.

## Controlled and Public Operation Reports

Method construction performs one complete scientific-state binding fingerprint, including one complete model fingerprint, outside the calculation-scoped report.

The deterministic nonzero RHF energy-gradient-descriptor sequence reports:

| Workflow | Complete scientific-state fingerprints | Complete model fingerprints | Public model-value fingerprints | Cheap evidence validations | Conservative cache-state fingerprints | Descriptor evaluations | Model forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Controlled package workflow | 2 | 2 | 0 | 0 | 0 | 1 | 1 |
| Public user-interruptible workflow | 2 | 4 | 2 | 2 | 2 | 1 | 1 |

## Consolidated Source Structure

The twelve consolidated owners named by the requirements each have one module docstring, one initial import section, unique imported symbols, and at most one final `__all__` assignment. RHF, RKS, UHF, and UKS direct-gradient and scalar-adjoint drivers inherit `GradientDriver.__init__` and receive backend options through its shared `options` contract.

Direct-gradient and scalar-adjoint algorithms remain in separate modules, audit implementations remain lazily imported, and the compatibility facade exports retain their supported public surfaces.

The current `deepks/deephf/` inventory is:

| Metric | Result |
| --- | ---: |
| Python files | 41 |
| Top-level Python files | 32 |
| Audit Python files | 9 |
| Total physical lines | 17,451 |
| Maximum module size | 908 lines in `pyscf_dft_provenance.py` |
| Maximum function size | 198 lines in `pyscf_rks_adjoint.py:_solve` |

## Regression Coverage

State-safety tests prove boundary-time rejection for tracked mutation, `.data` parameter mutation, writable shared NumPy buffer mutation, parameter replacement, and each trusted thermal helper replacement. The untracked value tests compare updated independent energy and gradient calculations with their original nonzero results and verify transaction publisher cleanup.

Architecture tests enforce the 41-file package limit, consolidated-module document and import structure, unique imports, final export ownership, inherited gradient constructors, direct-versus-adjoint separation, lazy audit loading, and existing size limits.

## Verification

The focused state-safety, performance, and architecture objectives passed with 96 tests.

The RHF, RKS, UHF, and UKS direct and scalar-adjoint scientific objectives passed with 618 tests.

The locked Python 3.11 environment synchronized with `uv sync --locked --python 3.11`, and the complete suite passed with 893 tests using `uv run pytest -q`.

Source distribution and wheel construction passed with `uv build`.

`git diff --check` passed.
