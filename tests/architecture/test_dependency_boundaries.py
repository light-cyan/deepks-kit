import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_method_neutral_descriptor_does_not_depend_on_deephf_or_model():
    for path in (ROOT / "deepks" / "descriptor").glob("*.py"):
        imports = imported_modules(path)
        assert not any(name.startswith("deepks.deephf") for name in imports)
        assert not any(name.startswith("deepks.model") for name in imports)


def test_generic_adjoint_algebra_is_independent_of_pyscf_and_workflows():
    imports = imported_modules(ROOT / "deepks" / "deephf" / "adjoint.py")
    assert not any(name.startswith("pyscf") for name in imports)
    assert not any(name.startswith("deepks.model") for name in imports)
    assert not any(name.startswith("deepks.deephf.workflow") for name in imports)
    assert not any(name.startswith("deepks.deephf.force_data") for name in imports)


def test_family_modules_do_not_depend_on_other_family_drivers():
    forbidden = {
        "pyscf_rhf": ("pyscf_uhf", "pyscf_rks", "pyscf_uks"),
        "pyscf_uhf": ("pyscf_rhf", "pyscf_rks", "pyscf_uks"),
        "pyscf_rks": ("pyscf_uhf", "pyscf_uks"),
    }
    directory = ROOT / "deepks" / "deephf"
    for prefix, other_families in forbidden.items():
        for path in directory.glob(f"{prefix}_*.py"):
            imports = imported_modules(path)
            assert not any(
                any(family in name for family in other_families)
                for name in imports
            ), path


def test_internal_modules_do_not_import_public_compatibility_facades():
    directory = ROOT / "deepks" / "deephf"
    facades = {"pyscf_rhf", "pyscf_uhf", "pyscf_rks", "pyscf_uks"}
    facade_files = {f"{name}.py" for name in facades}
    for path in directory.rglob("*.py"):
        if path.name == "__init__.py" or path.name in facade_files:
            continue
        assert facades.isdisjoint(imported_modules(path)), path
