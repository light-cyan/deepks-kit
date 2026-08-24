import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
PRODUCTION_LIMIT = 1000
CALCULATION_FUNCTION_LIMIT = 199
MODULE_COUNT_LIMIT = 42
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


def test_deephf_package_has_a_bounded_aggregate_topology():
    directory = ROOT / "deepks" / "deephf"
    modules = tuple(directory.rglob("*.py"))
    top_level = tuple(directory.glob("*.py"))
    audits = tuple((directory / "audits").glob("*.py"))
    assert len(modules) <= MODULE_COUNT_LIMIT
    assert len(top_level) <= 33
    assert len(audits) <= 9


def test_compatibility_facades_contain_no_class_or_function_implementation():
    directory = ROOT / "deepks" / "deephf"
    for name in FACADE_NAMES:
        tree = ast.parse((directory / name).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef))
            for node in tree.body
        )


def test_compatibility_facades_have_explicit_exports_without_private_aliases():
    directory = ROOT / "deepks" / "deephf"
    for name in FACADE_NAMES:
        tree = ast.parse((directory / name).read_text(encoding="utf-8"))
        exported = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        ]
        assert len(exported) == 1
        assert not any(isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names) for node in tree.body)
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any((alias.asname or alias.name.rsplit(".", 1)[-1]).startswith("_") for alias in node.names)
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
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
            size = node.end_lineno - start + 1
            if size > CALCULATION_FUNCTION_LIMIT:
                module = path.relative_to(directory)
                oversized[f"{module}:{node.name}"] = size
    assert oversized == {}


def test_production_functions_have_no_long_exact_ast_duplicates():
    directory = ROOT / "deepks" / "deephf"
    implementations = {}
    duplicates = {}
    for path in directory.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or len(node.body) <= 4:
                continue
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
            location = f"{path.relative_to(directory)}:{node.name}"
            if body in implementations:
                duplicates.setdefault(implementations[body], []).append(location)
            else:
                implementations[body] = location
    assert duplicates == {}


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


def test_dense_audits_are_not_imported_at_module_load_time():
    directory = ROOT / "deepks" / "deephf"
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        eager_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                (isinstance(node, ast.ImportFrom) and node.module and "audits" in node.module.split("."))
                or (isinstance(node, ast.Import) and any("audits" in alias.name.split(".") for alias in node.names))
            )
        ]
        assert eager_imports == [], path


def test_corrnet_setters_do_not_bypass_torch_mutation_versions():
    tree = ast.parse((ROOT / "deepks" / "model" / "model.py").read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Attribute) and node.attr == "data" for node in ast.walk(tree))


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
