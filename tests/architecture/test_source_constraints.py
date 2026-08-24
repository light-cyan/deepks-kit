import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
PRODUCTION_LIMIT = 1200
CALCULATION_FUNCTION_LIMIT = 199
FACADE_NAMES = {"pyscf_rhf.py", "pyscf_uhf.py", "pyscf_rks.py", "pyscf_uks.py"}
AUDIT_ORCHESTRATORS = {
    "_audit_adjoint",
    "_audit_rks_reference",
    "_audit_uks_reference",
    "audit_adjoint",
    "audit_response_equations",
    "validate_reference",
    "validate_uhf_reference",
}


def test_deephf_production_modules_have_cohesive_sizes():
    directory = ROOT / "deepks" / "deephf"
    oversized = {
        str(path.relative_to(directory)): len(path.read_text(encoding="utf-8").splitlines())
        for path in directory.rglob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > PRODUCTION_LIMIT
    }
    assert oversized == {}


def test_compatibility_facades_contain_no_class_or_function_implementation():
    directory = ROOT / "deepks" / "deephf"
    for name in FACADE_NAMES:
        tree = ast.parse((directory / name).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef))
            for node in tree.body
        )


def test_production_calculation_functions_remain_below_hard_limit():
    directory = ROOT / "deepks" / "deephf"
    oversized = {}
    for path in directory.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = node.end_lineno - node.lineno + 1
            if size > CALCULATION_FUNCTION_LIMIT:
                module = path.relative_to(directory)
                oversized[f"{module}:{node.name}"] = size
    assert oversized == {}


def test_dense_audit_implementations_are_isolated_from_production_modules():
    directory = ROOT / "deepks" / "deephf"
    audit_entry_points = {
        "_audit_adjoint",
        "_audit_rks_reference",
        "_audit_uks_reference",
        "audit_adjoint",
        "audit_response_equations",
        "audit_rks_reference",
        "audit_uks_reference",
        "validate_response_operator_exact",
    }
    for path in directory.glob("pyscf_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in audit_entry_points:
                assert node.end_lineno - node.lineno + 1 <= 20, (path, node.name)


def test_audit_entry_points_only_orchestrate_responsibility_helpers():
    directory = ROOT / "deepks" / "deephf" / "audits"
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in AUDIT_ORCHESTRATORS:
                assert node.end_lineno - node.lineno + 1 <= 100, (path, node.name)


def test_architecture_modules_remain_focused():
    directory = ROOT / "tests" / "architecture"
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) < 500
        for path in directory.glob("test_*.py")
    )
