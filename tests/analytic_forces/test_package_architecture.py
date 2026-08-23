import ast
import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "deepks"


def _module_path(source_path):
    relative_path = source_path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = list(relative_path.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_path(source_path):
    module_parts = _module_path(source_path).split(".")
    if source_path.name == "__init__.py":
        return module_parts
    return module_parts[:-1]


def _literal_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left)
        right = _literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _resolved_import(source_path, module_name):
    if not module_name.startswith("."):
        return module_name
    level = len(module_name) - len(module_name.lstrip("."))
    retained_parts = len(_package_path(source_path)) - level + 1
    parts = _package_path(source_path)[:retained_parts]
    remainder = module_name[level:]
    if remainder:
        parts.extend(remainder.split("."))
    return ".".join(parts)


def _import_aliases(source_path, tree):
    aliases = {
        "__import__": "builtins.__import__",
        "getattr": "builtins.getattr",
        "hasattr": "builtins.hasattr",
        "setattr": "builtins.setattr",
        "delattr": "builtins.delattr",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                aliases[bound_name] = alias.name if alias.asname else bound_name
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_name = _resolved_import(
                source_path,
                "." * node.level + node.module if node.level else node.module,
            )
            for alias in node.names:
                if alias.name != "*":
                    aliases[alias.asname or alias.name] = (
                        f"{module_name}.{alias.name}"
                    )
    return aliases


def _resolved_name(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        prefix = _resolved_name(node.value, aliases)
        if prefix:
            return f"{prefix}.{node.attr}"
    return None


def _is_dynamic_import_call(node, aliases):
    function_name = _resolved_name(node.func, aliases)
    if function_name in {
        "builtins.__import__",
        "importlib.import_module",
    }:
        return True
    if isinstance(node.func, ast.Call):
        getter_name = _resolved_name(node.func.func, aliases)
        getter_args = node.func.args
        return (
            getter_name == "builtins.getattr"
            and len(getter_args) >= 2
            and _resolved_name(getter_args[0], aliases) == "importlib"
            and _literal_string(getter_args[1]) == "import_module"
        )
    return False


def _imported_modules(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    package_parts = _package_path(source_path)
    aliases = _import_aliases(source_path, tree)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                retained_parts = len(package_parts) - node.level + 1
                target_parts = package_parts[:retained_parts]
                if node.module:
                    target_parts.extend(node.module.split("."))
                module_name = ".".join(target_parts)
                imported.append(module_name)
                imported.extend(
                    f"{module_name}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            elif node.module:
                imported.append(node.module)
                imported.extend(
                    f"{node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
        elif (
            isinstance(node, ast.Call)
            and node.args
            and _is_dynamic_import_call(node, aliases)
        ):
            module_name = _literal_string(node.args[0])
            if module_name is not None:
                imported.append(_resolved_import(source_path, module_name))
    return sorted(set(imported))


def _imports_prefix(module_name, package_name):
    return module_name == package_name or module_name.startswith(package_name + ".")


def _dependency_violations(package_directory, forbidden_packages):
    violations = []
    for source_path in sorted(package_directory.rglob("*.py")):
        for imported_module in _imported_modules(source_path):
            if any(
                _imports_prefix(imported_module, forbidden_package)
                for forbidden_package in forbidden_packages
            ):
                violations.append(
                    (
                        str(source_path.relative_to(REPOSITORY_ROOT)),
                        imported_module,
                    )
                )
    return violations


def _pyscf_response_imports(source_path):
    return [
        module_name
        for module_name in _imported_modules(source_path)
        if module_name == "pyscf.scf.cphf"
        or module_name.startswith("pyscf.scf.cphf.")
        or module_name == "pyscf.scf.ucphf"
        or module_name.startswith("pyscf.scf.ucphf.")
        or module_name == "pyscf.hessian.rhf"
        or module_name.startswith("pyscf.hessian.rhf.")
        or module_name == "pyscf.hessian.uhf"
        or module_name.startswith("pyscf.hessian.uhf.")
        or module_name == "pyscf.hessian.rks"
        or module_name.startswith("pyscf.hessian.rks.")
        or module_name == "pyscf.grad.rks"
        or module_name.startswith("pyscf.grad.rks.")
        or module_name == "pyscf.hessian.uks"
        or module_name.startswith("pyscf.hessian.uks.")
        or module_name == "pyscf.grad.uks"
        or module_name.startswith("pyscf.grad.uks.")
        or module_name == "pyscf.dft.numint"
        or module_name.startswith("pyscf.dft.numint.")
        or module_name == "pyscf.dft.libxc"
        or module_name.startswith("pyscf.dft.libxc.")
        or module_name == "pyscf.dft.gen_grid"
        or module_name.startswith("pyscf.dft.gen_grid.")
        or module_name == "pyscf.dft.radi"
        or module_name.startswith("pyscf.dft.radi.")
    ]


def _private_attribute_accesses(source_path, attribute_names):
    tree = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    aliases = _import_aliases(source_path, tree)
    direct_accesses = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in attribute_names
    ]
    dynamic_accesses = []
    attribute_functions = {
        "builtins.getattr",
        "builtins.hasattr",
        "builtins.setattr",
        "builtins.delattr",
        "object.__getattribute__",
        "operator.attrgetter",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = _resolved_name(node.func, aliases)
            if function_name in attribute_functions or (
                function_name is not None
                and function_name.endswith(".__getattribute__")
            ):
                for argument in (*node.args, *(item.value for item in node.keywords)):
                    attribute_name = _literal_string(argument)
                    if attribute_name in attribute_names:
                        dynamic_accesses.append(attribute_name)
            if function_name is not None and function_name.endswith(".get"):
                for argument in node.args:
                    attribute_name = _literal_string(argument)
                    if attribute_name in attribute_names:
                        dynamic_accesses.append(attribute_name)
        elif isinstance(node, ast.Subscript):
            attribute_name = _literal_string(node.slice)
            if attribute_name in attribute_names:
                dynamic_accesses.append(attribute_name)
    return sorted(set(direct_accesses + dynamic_accesses))


def _pyscf_compatibility_accesses(source_path):
    tree = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    aliases = _import_aliases(source_path, tree)
    names = set(_imported_modules(source_path))
    names.update(
        resolved
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
        and (resolved := _resolved_name(node, aliases)) is not None
    )
    facilities = set()
    for name in names:
        if _imports_prefix(name, "pyscf.scf.cphf"):
            facilities.add("cphf")
        if _imports_prefix(name, "pyscf.scf.ucphf"):
            facilities.add("ucphf")
        if _imports_prefix(name, "pyscf.hessian.rhf"):
            facilities.add("rhf_hessian")
        if _imports_prefix(name, "pyscf.hessian.uhf"):
            facilities.add("uhf_hessian")
        if _imports_prefix(name, "pyscf.hessian.rks"):
            facilities.add("rks_hessian")
        if _imports_prefix(name, "pyscf.grad.rks"):
            facilities.add("rks_gradient")
        if _imports_prefix(name, "pyscf.hessian.uks"):
            facilities.add("uks_hessian")
        if _imports_prefix(name, "pyscf.grad.uks"):
            facilities.add("uks_gradient")
        if _imports_prefix(name, "pyscf.dft.numint"):
            facilities.add("rks_numint")
        if _imports_prefix(name, "pyscf.dft.libxc"):
            facilities.add("rks_libxc")
        if (
            _imports_prefix(name, "pyscf.dft.gen_grid")
            or _imports_prefix(name, "pyscf.dft.radi")
        ):
            facilities.add("rks_grid")
        if _imports_prefix(name, "pyscf.dft.rks"):
            facilities.add("rks_reference")
        if _imports_prefix(name, "pyscf.dft.uks"):
            facilities.add("uks_reference")
        if name.startswith("pyscf.") and "intor_cross" in name.split("."):
            facilities.add("cross_overlap")
    return sorted(facilities)


def _symbol_accesses(source_path, nodes, symbol_names):
    tree = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    aliases = _import_aliases(source_path, tree)
    accesses = set()
    for root in nodes(tree):
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and node.id in symbol_names:
                accesses.add(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in symbol_names:
                accesses.add(node.attr)
            elif (
                isinstance(node, ast.alias)
                and node.name.split(".")[-1] in symbol_names
            ):
                accesses.add(node.name.split(".")[-1])
            elif isinstance(node, (ast.Call, ast.Subscript)):
                candidates = []
                if isinstance(node, ast.Call):
                    candidates.extend(node.args)
                    candidates.extend(item.value for item in node.keywords)
                else:
                    candidates.append(node.slice)
                for candidate in candidates:
                    symbol_name = _literal_string(candidate)
                    if symbol_name in symbol_names:
                        accesses.add(symbol_name)
            if isinstance(node, ast.Call):
                function_name = _resolved_name(node.func, aliases)
                if function_name in {
                    "builtins.getattr",
                    "operator.attrgetter",
                }:
                    for argument in node.args:
                        symbol_name = _literal_string(argument)
                        if symbol_name in symbol_names:
                            accesses.add(symbol_name)
    return sorted(accesses)


def test_package_dependency_directions_are_one_way():
    boundaries = {
        "descriptor": (
            "deepks.deepks",
            "deepks.deephf",
            "deepks.data",
            "deepks.model",
        ),
        "deepks": ("deepks.deephf",),
        "deephf": ("deepks.deepks",),
        "data": ("deepks.deepks", "deepks.deephf"),
        "model": ("deepks.deepks", "deepks.deephf"),
    }

    violations = {
        package_name: _dependency_violations(
            PACKAGE_ROOT / package_name,
            forbidden_packages,
        )
        for package_name, forbidden_packages in boundaries.items()
    }

    assert violations == {package_name: [] for package_name in boundaries}


def test_legacy_scf_package_has_no_import_spec():
    assert importlib.util.find_spec("deepks.scf") is None


def test_pyscf_response_imports_are_isolated_in_the_compatibility_adapter():
    response_imports = []
    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative_path = str(source_path.relative_to(REPOSITORY_ROOT))
        for module_name in _pyscf_response_imports(source_path):
            is_rks_facility = any(
                _imports_prefix(module_name, prefix)
                for prefix in (
                    "pyscf.hessian.rks",
                    "pyscf.grad.rks",
                    "pyscf.dft.numint",
                    "pyscf.dft.libxc",
                    "pyscf.dft.gen_grid",
                    "pyscf.dft.radi",
                    "pyscf.hessian.uks",
                    "pyscf.grad.uks",
                )
            )
            if is_rks_facility and source_path.parent != PACKAGE_ROOT / "deephf":
                continue
            response_imports.append((relative_path, module_name))

    owners = {}
    for source_path, module_name in response_imports:
        if _imports_prefix(module_name, "pyscf.scf.cphf"):
            facility = "cphf"
        elif _imports_prefix(module_name, "pyscf.scf.ucphf"):
            facility = "ucphf"
        elif _imports_prefix(module_name, "pyscf.hessian.rhf"):
            facility = "rhf_hessian"
        elif _imports_prefix(module_name, "pyscf.hessian.uhf"):
            facility = "uhf_hessian"
        elif _imports_prefix(module_name, "pyscf.hessian.rks"):
            facility = "rks_hessian"
        elif _imports_prefix(module_name, "pyscf.grad.rks"):
            facility = "rks_gradient"
        elif _imports_prefix(module_name, "pyscf.hessian.uks"):
            facility = "uks_hessian"
        elif _imports_prefix(module_name, "pyscf.grad.uks"):
            facility = "uks_gradient"
        elif _imports_prefix(module_name, "pyscf.dft.numint"):
            facility = "rks_numint"
        elif _imports_prefix(module_name, "pyscf.dft.libxc"):
            facility = "rks_libxc"
        elif (
            _imports_prefix(module_name, "pyscf.dft.gen_grid")
            or _imports_prefix(module_name, "pyscf.dft.radi")
        ):
            facility = "rks_grid"
        else:
            raise AssertionError(f"unclassified response import {module_name}")
        owners.setdefault(facility, set()).add(source_path)

    assert owners == {
        "cphf": {
            "deepks/deephf/pyscf_rhf.py",
            "deepks/deephf/pyscf_rks.py",
        },
        "rhf_hessian": {"deepks/deephf/pyscf_rhf.py"},
        "ucphf": {"deepks/deephf/pyscf_uhf.py"},
        "uhf_hessian": {"deepks/deephf/pyscf_uhf.py"},
        "rks_hessian": {"deepks/deephf/pyscf_rks.py"},
        "rks_gradient": {"deepks/deephf/pyscf_rks.py"},
        "uks_hessian": {"deepks/deephf/pyscf_uks.py"},
        "uks_gradient": {"deepks/deephf/pyscf_uks.py"},
        "rks_numint": {"deepks/deephf/pyscf_rks.py", "deepks/deephf/pyscf_uks.py"},
        "rks_libxc": {"deepks/deephf/pyscf_rks.py", "deepks/deephf/pyscf_uks.py"},
        "rks_grid": {"deepks/deephf/pyscf_rks.py"},
    }
    imported_modules = {module_name for _, module_name in response_imports}
    assert "pyscf.scf.cphf" in imported_modules
    assert "pyscf.hessian.rhf" in imported_modules
    assert "pyscf.scf.ucphf" in imported_modules
    assert "pyscf.hessian.uhf" in imported_modules
    assert "pyscf.hessian.rks" in imported_modules
    assert "pyscf.grad.rks" in imported_modules
    assert "pyscf.hessian.uks" in imported_modules
    assert "pyscf.grad.uks" in imported_modules
    assert "pyscf.dft.numint" in imported_modules
    assert "pyscf.dft.libxc" in imported_modules
    assert "pyscf.dft.gen_grid" in imported_modules
    assert "pyscf.dft.radi" in imported_modules


def test_reference_neutral_adjoint_has_no_reference_or_application_imports():
    source_path = PACKAGE_ROOT / "deephf" / "adjoint.py"
    forbidden_packages = (
        "pyscf",
        "deepks.deepks",
        "deepks.data",
        "deepks.model",
        "deepks.deephf.method",
        "deepks.deephf.pyscf_rhf",
    )
    violations = [
        module_name
        for module_name in _imported_modules(source_path)
        if any(
            _imports_prefix(module_name, package_name)
            for package_name in forbidden_packages
        )
    ]

    assert violations == []


def test_deephf_pyscf_compatibility_facilities_have_one_adapter_owner():
    accesses = []
    for source_path in sorted((PACKAGE_ROOT / "deephf").rglob("*.py")):
        relative_path = str(source_path.relative_to(REPOSITORY_ROOT))
        accesses.extend(
            (relative_path, facility)
            for facility in _pyscf_compatibility_accesses(source_path)
        )

    owners = {}
    for source_path, facility in accesses:
        owners.setdefault(facility, set()).add(source_path)

    assert owners == {
        "cphf": {
            "deepks/deephf/pyscf_rhf.py",
            "deepks/deephf/pyscf_rks.py",
        },
        "cross_overlap": {"deepks/deephf/pyscf_rhf.py"},
        "rhf_hessian": {"deepks/deephf/pyscf_rhf.py"},
        "ucphf": {"deepks/deephf/pyscf_uhf.py"},
        "uhf_hessian": {"deepks/deephf/pyscf_uhf.py"},
        "rks_hessian": {"deepks/deephf/pyscf_rks.py"},
        "rks_gradient": {"deepks/deephf/pyscf_rks.py"},
        "uks_hessian": {"deepks/deephf/pyscf_uks.py"},
        "uks_gradient": {"deepks/deephf/pyscf_uks.py"},
        "rks_numint": {"deepks/deephf/pyscf_rks.py", "deepks/deephf/pyscf_uks.py"},
        "rks_libxc": {"deepks/deephf/pyscf_rks.py", "deepks/deephf/pyscf_uks.py"},
        "rks_grid": {"deepks/deephf/pyscf_rks.py"},
        "rks_reference": {"deepks/deephf/pyscf_rks.py", "deepks/deephf/workflow.py"},
        "uks_reference": {"deepks/deephf/pyscf_uks.py", "deepks/deephf/workflow.py"},
    }


def test_deephf_nonadapter_modules_do_not_access_private_molecular_state():
    private_state = {
        "_atm",
        "_bas",
        "_env",
        "_basis",
        "_ecp",
        "_eri",
        "_pseudo",
        "_numint",
        "__dict__",
    }
    guarded_modules = (
        "capabilities.py",
        "scanner.py",
        "method.py",
        "zvector.py",
        "force_data.py",
        "uhf_method.py",
        "uhf_gradient.py",
        "uhf_zvector.py",
        "rks_method.py",
        "rks_gradient.py",
        "rks_zvector.py",
        "uks_method.py",
        "uks_gradient.py",
        "uks_zvector.py",
    )
    violations = {
        module_name: _private_attribute_accesses(
            PACKAGE_ROOT / "deephf" / module_name,
            private_state,
        )
        for module_name in guarded_modules
    }

    assert violations == {module_name: [] for module_name in guarded_modules}


def test_rks_semiprivate_pyscf_facilities_have_one_adapter_owner():
    semiprivate_symbols = {
        "_CUSTOM_FUNC_R",
        "_itrf",
        "grids_response_cc",
    }

    def whole_module(tree):
        return (tree,)

    owners = {symbol: set() for symbol in semiprivate_symbols}
    for source_path in sorted((PACKAGE_ROOT / "deephf").rglob("*.py")):
        relative_path = str(source_path.relative_to(REPOSITORY_ROOT))
        for symbol in _symbol_accesses(
            source_path,
            whole_module,
            semiprivate_symbols,
        ):
            owners[symbol].add(relative_path)

    assert owners == {
        "_CUSTOM_FUNC_R": {"deepks/deephf/pyscf_rks.py", "deepks/deephf/pyscf_uks.py"},
        "_itrf": {"deepks/deephf/pyscf_rks.py"},
        "grids_response_cc": {"deepks/deephf/pyscf_rks.py"},
    }


def test_capabilities_is_pyscf_neutral_and_reference_validation_has_exact_owners():
    capabilities_path = PACKAGE_ROOT / "deephf" / "capabilities.py"
    pyscf_imports = [
        module_name
        for module_name in _imported_modules(capabilities_path)
        if _imports_prefix(module_name, "pyscf")
    ]
    owners = {
        "validate_reference": [],
        "validate_uhf_reference": [],
        "validate_rks_reference": [],
        "validate_uks_reference": [],
    }
    for source_path in sorted((PACKAGE_ROOT / "deephf").rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in owners
            ):
                owners[node.name].append(
                    str(source_path.relative_to(REPOSITORY_ROOT))
                )

    assert pyscf_imports == []
    assert owners == {
        "validate_reference": ["deepks/deephf/pyscf_rhf.py"],
        "validate_uhf_reference": ["deepks/deephf/pyscf_uhf.py"],
        "validate_rks_reference": ["deepks/deephf/pyscf_rks.py"],
        "validate_uks_reference": ["deepks/deephf/pyscf_uks.py"],
    }


def test_zvector_path_has_no_direct_response_symbol_access():
    forbidden_symbols = {
        "RHFDeePHFGradients",
        "RHFResponseAdapter",
        "response",
        "first_order_density",
        "dq_dR_response",
        "dq_dR_relaxed",
    }
    zvector_path = PACKAGE_ROOT / "deephf" / "zvector.py"
    method_path = PACKAGE_ROOT / "deephf" / "method.py"

    def whole_module(tree):
        return (tree,)

    def method_zvector_nodes(tree):
        return tuple(
            node
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef) and class_node.name == "DeePHF"
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"adjoint", "_zvector_inputs"}
        )

    violations = {
        "deepks/deephf/zvector.py": _symbol_accesses(
            zvector_path,
            whole_module,
            forbidden_symbols,
        ),
        "deepks/deephf/method.py:adjoint": _symbol_accesses(
            method_path,
            method_zvector_nodes,
            forbidden_symbols,
        ),
    }

    assert violations == {name: [] for name in violations}


def test_rhf_zvector_scanner_and_force_data_do_not_access_non_rhf_symbols():
    forbidden_symbols = {
        "UHFDeePHF",
        "UHFDeePHFGradients",
        "UHFDeePHFZVectorGradients",
        "UHFAdjoint",
        "UHFAdjointAdapter",
        "UHFResponse",
        "UHFResponseAdapter",
        "UHFResponseDiagnostics",
        "validate_uhf_reference",
        "RKSDeePHF",
        "RKSDeePHFGradients",
        "RKSDeePHFZVectorGradients",
        "RKSAdjoint",
        "RKSAdjointAdapter",
        "RKSResponse",
        "RKSResponseAdapter",
        "RKSResponseDiagnostics",
        "validate_rks_reference",
        "UKSDeePHF",
        "UKSDeePHFGradients",
        "UKSDeePHFZVectorGradients",
        "UKSAdjoint",
        "UKSAdjointAdapter",
        "UKSResponse",
        "UKSResponseAdapter",
        "UKSResponseDiagnostics",
        "validate_uks_reference",
    }

    def whole_module(tree):
        return (tree,)

    guarded_modules = ("zvector.py", "scanner.py", "force_data.py")
    violations = {
        module_name: _symbol_accesses(
            PACKAGE_ROOT / "deephf" / module_name,
            whole_module,
            forbidden_symbols,
        )
        for module_name in guarded_modules
    }

    assert violations == {module_name: [] for module_name in guarded_modules}


def test_uhf_paths_do_not_access_other_reference_backends_or_force_data_symbols():
    forbidden_symbols = {
        "RHFAdjoint",
        "RHFAdjointAdapter",
        "RHFDeePHFGradientScanner",
        "RHFDeePHFZVectorGradients",
        "RHFForceFrame",
        "generate_rhf_force_frame",
        "write_rhf_force_dataset",
        "RKSDeePHF",
        "RKSDeePHFGradients",
        "RKSDeePHFZVectorGradients",
        "RKSAdjoint",
        "RKSAdjointAdapter",
        "RKSResponse",
        "RKSResponseAdapter",
        "RKSResponseDiagnostics",
        "validate_rks_reference",
        "UKSDeePHF",
        "UKSDeePHFGradients",
        "UKSDeePHFZVectorGradients",
        "UKSAdjoint",
        "UKSAdjointAdapter",
        "UKSResponse",
        "UKSResponseAdapter",
        "validate_uks_reference",
    }

    def whole_module(tree):
        return (tree,)

    guarded_modules = (
        "pyscf_uhf.py",
        "uhf_method.py",
        "uhf_gradient.py",
        "uhf_zvector.py",
    )
    violations = {
        module_name: _symbol_accesses(
            PACKAGE_ROOT / "deephf" / module_name,
            whole_module,
            forbidden_symbols,
        )
        for module_name in guarded_modules
    }

    assert violations == {module_name: [] for module_name in guarded_modules}


def test_rks_paths_do_not_access_other_reference_backends():
    forbidden_symbols = {
        "RHFAdjoint",
        "RHFAdjointAdapter",
        "RHFDeePHFGradientScanner",
        "RHFDeePHFZVectorGradients",
        "RHFForceFrame",
        "UHFDeePHF",
        "UHFDeePHFGradients",
        "UHFDeePHFZVectorGradients",
        "UHFAdjoint",
        "UHFAdjointAdapter",
        "UHFResponse",
        "UHFResponseAdapter",
        "generate_rhf_force_frame",
        "write_rhf_force_dataset",
        "UKSDeePHF",
        "UKSDeePHFGradients",
        "UKSDeePHFZVectorGradients",
        "UKSAdjoint",
        "UKSAdjointAdapter",
        "UKSResponse",
        "UKSResponseAdapter",
        "validate_uks_reference",
    }

    def whole_module(tree):
        return (tree,)

    guarded_modules = (
        "pyscf_rks.py",
        "rks_method.py",
        "rks_gradient.py",
        "rks_zvector.py",
    )
    violations = {
        module_name: _symbol_accesses(
            PACKAGE_ROOT / "deephf" / module_name,
            whole_module,
            forbidden_symbols,
        )
        for module_name in guarded_modules
    }

    assert violations == {module_name: [] for module_name in guarded_modules}


def test_rks_zvector_path_has_no_direct_response_symbol_access():
    forbidden_symbols = {
        "RKSDeePHFGradients",
        "RKSResponseAdapter",
        "response",
        "first_order_density",
        "dq_dR_response",
        "dq_dR_relaxed",
    }
    zvector_path = PACKAGE_ROOT / "deephf" / "rks_zvector.py"
    method_path = PACKAGE_ROOT / "deephf" / "rks_method.py"

    def whole_module(tree):
        return (tree,)

    def method_zvector_nodes(tree):
        return tuple(
            node
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef)
            and class_node.name == "RKSDeePHF"
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {
                "adjoint",
                "_zvector_inputs",
            }
        )

    violations = {
        "deepks/deephf/rks_zvector.py": _symbol_accesses(
            zvector_path,
            whole_module,
            forbidden_symbols,
        ),
        "deepks/deephf/rks_method.py:adjoint": _symbol_accesses(
            method_path,
            method_zvector_nodes,
            forbidden_symbols,
        ),
    }

    assert violations == {name: [] for name in violations}


def test_uks_zvector_path_has_no_direct_response_symbol_access():
    forbidden_symbols = {
        "UKSDeePHFGradients",
        "UKSResponseAdapter",
        "response",
        "first_order_density",
        "first_order_spin_density",
        "dq_dR_response",
        "dq_dR_response_spin",
        "dq_dR_relaxed",
        "dq_dR_relaxed_spin",
    }

    def whole_module(tree):
        return (tree,)

    def method_zvector_nodes(tree):
        return tuple(
            node
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef) and class_node.name == "UKSDeePHF"
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"adjoint", "_zvector_inputs"}
        )

    violations = {
        "deepks/deephf/uks_zvector.py": _symbol_accesses(PACKAGE_ROOT / "deephf" / "uks_zvector.py", whole_module, forbidden_symbols),
        "deepks/deephf/uks_method.py:adjoint": _symbol_accesses(PACKAGE_ROOT / "deephf" / "uks_method.py", method_zvector_nodes, forbidden_symbols),
    }
    assert violations == {name: [] for name in violations}


def test_uhf_zvector_path_has_no_direct_response_symbol_access():
    forbidden_symbols = {
        "UHFDeePHFGradients",
        "UHFResponseAdapter",
        "response",
        "first_order_density",
        "first_order_spin_density",
        "dq_dR_response",
        "dq_dR_response_spin",
        "dq_dR_relaxed",
        "dq_dR_relaxed_spin",
    }
    zvector_path = PACKAGE_ROOT / "deephf" / "uhf_zvector.py"
    method_path = PACKAGE_ROOT / "deephf" / "uhf_method.py"

    def whole_module(tree):
        return (tree,)

    def method_zvector_nodes(tree):
        return tuple(
            node
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef)
            and class_node.name == "UHFDeePHF"
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {
                "adjoint",
                "_zvector_inputs",
            }
        )

    violations = {
        "deepks/deephf/uhf_zvector.py": _symbol_accesses(
            zvector_path,
            whole_module,
            forbidden_symbols,
        ),
        "deepks/deephf/uhf_method.py:adjoint": _symbol_accesses(
            method_path,
            method_zvector_nodes,
            forbidden_symbols,
        ),
    }

    assert violations == {name: [] for name in violations}


def test_force_data_does_not_access_pyscf_private_basis_metadata():
    source_path = PACKAGE_ROOT / "deephf" / "force_data.py"

    assert _private_attribute_accesses(source_path, {"_basis", "_ecp"}) == []


def test_shared_descriptor_symbols_have_one_module_level_owner():
    expected_owners = {
        "AtomicDensityDescriptor": "deepks/descriptor/projection.py",
        "batch_jacobian": "deepks/descriptor/core.py",
        "build_projector_molecule": "deepks/descriptor/projection.py",
        "dD_dR_explicit": "deepks/descriptor/derivatives.py",
        "descriptor": "deepks/descriptor/core.py",
        "descriptor_atom_indices": "deepks/descriptor/projection.py",
        "dq_dP": "deepks/descriptor/core.py",
        "dq_dR_explicit": "deepks/descriptor/derivatives.py",
        "is_ghost_atom": "deepks/descriptor/projection.py",
        "occupied_virtual_gradient": "deepks/descriptor/orbitals.py",
        "projected_density": "deepks/descriptor/core.py",
        "shell_eigenvalues": "deepks/descriptor/core.py",
    }
    actual_owners = {symbol: [] for symbol in expected_owners}

    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        relative_path = str(source_path.relative_to(REPOSITORY_ROOT))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in actual_owners:
                    actual_owners[node.name].append(relative_path)

    assert actual_owners == {
        symbol: [owner] for symbol, owner in expected_owners.items()
    }


def test_zvector_result_symbols_have_one_module_level_owner():
    expected_owners = {
        "AdjointDiagnostics": "deepks/deephf/adjoint.py",
        "AdjointResult": "deepks/deephf/adjoint.py",
        "RHFAdjointDiagnostics": "deepks/deephf/pyscf_rhf.py",
        "RHFAdjoint": "deepks/deephf/pyscf_rhf.py",
        "RHFDeePHFZVectorGradients": "deepks/deephf/zvector.py",
    }
    actual_owners = {symbol: [] for symbol in expected_owners}

    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        relative_path = str(source_path.relative_to(REPOSITORY_ROOT))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in actual_owners:
                    actual_owners[node.name].append(relative_path)

    assert actual_owners == {
        symbol: [owner] for symbol, owner in expected_owners.items()
    }


def test_uhf_result_symbols_have_one_module_level_owner():
    expected_owners = {
        "UHFDeePHF": "deepks/deephf/uhf_method.py",
        "UHFDeePHFGradients": "deepks/deephf/uhf_gradient.py",
        "UHFDeePHFZVectorGradients": "deepks/deephf/uhf_zvector.py",
        "UHFAdjoint": "deepks/deephf/pyscf_uhf.py",
        "UHFAdjointAdapter": "deepks/deephf/pyscf_uhf.py",
        "UHFAdjointDiagnostics": "deepks/deephf/pyscf_uhf.py",
        "UHFAdjointError": "deepks/deephf/pyscf_uhf.py",
        "UHFResponse": "deepks/deephf/pyscf_uhf.py",
        "UHFResponseAdapter": "deepks/deephf/pyscf_uhf.py",
        "UHFResponseDiagnostics": "deepks/deephf/pyscf_uhf.py",
        "UHFResponseError": "deepks/deephf/pyscf_uhf.py",
    }
    actual_owners = {symbol: [] for symbol in expected_owners}

    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        relative_path = str(source_path.relative_to(REPOSITORY_ROOT))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in actual_owners:
                    actual_owners[node.name].append(relative_path)

    assert actual_owners == {
        symbol: [owner] for symbol, owner in expected_owners.items()
    }


def test_rks_direct_result_symbols_have_one_module_level_owner():
    expected_owners = {
        "RKSDeePHF": "deepks/deephf/rks_method.py",
        "RKSDeePHFGradients": "deepks/deephf/rks_gradient.py",
        "RKSDeePHFZVectorGradients": "deepks/deephf/rks_zvector.py",
        "RKSAdjoint": "deepks/deephf/pyscf_rks.py",
        "RKSAdjointAdapter": "deepks/deephf/pyscf_rks.py",
        "RKSAdjointDiagnostics": "deepks/deephf/pyscf_rks.py",
        "RKSFunctionalProvenance": "deepks/deephf/pyscf_rks.py",
        "RKSGridProvenance": "deepks/deephf/pyscf_rks.py",
        "RKSNativeGradient": "deepks/deephf/pyscf_rks.py",
        "RKSResponse": "deepks/deephf/pyscf_rks.py",
        "RKSResponseAdapter": "deepks/deephf/pyscf_rks.py",
        "RKSResponseDiagnostics": "deepks/deephf/pyscf_rks.py",
        "RKSResponseError": "deepks/deephf/pyscf_rks.py",
        "native_rks_gradient": "deepks/deephf/pyscf_rks.py",
        "rks_adjoint_integrity_fingerprint": "deepks/deephf/pyscf_rks.py",
        "rks_reference_fingerprint": "deepks/deephf/pyscf_rks.py",
        "rks_response_integrity_fingerprint": "deepks/deephf/pyscf_rks.py",
    }
    actual_owners = {symbol: [] for symbol in expected_owners}

    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        relative_path = str(source_path.relative_to(REPOSITORY_ROOT))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in actual_owners:
                    actual_owners[node.name].append(relative_path)

    assert actual_owners == {
        symbol: [owner] for symbol, owner in expected_owners.items()
    }


def test_uks_result_symbols_have_one_module_level_owner():
    expected_owners = {
        "UKSDeePHF": "deepks/deephf/uks_method.py",
        "UKSDeePHFGradients": "deepks/deephf/uks_gradient.py",
        "UKSDeePHFZVectorGradients": "deepks/deephf/uks_zvector.py",
        "UKSAdjoint": "deepks/deephf/pyscf_uks.py",
        "UKSAdjointAdapter": "deepks/deephf/pyscf_uks.py",
        "UKSAdjointDiagnostics": "deepks/deephf/pyscf_uks.py",
        "UKSAdjointError": "deepks/deephf/pyscf_uks.py",
        "UKSNativeGradient": "deepks/deephf/pyscf_uks.py",
        "UKSResponse": "deepks/deephf/pyscf_uks.py",
        "UKSResponseAdapter": "deepks/deephf/pyscf_uks.py",
        "UKSResponseDiagnostics": "deepks/deephf/pyscf_uks.py",
        "UKSResponseError": "deepks/deephf/pyscf_uks.py",
    }
    actual_owners = {symbol: [] for symbol in expected_owners}
    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        relative_path = str(source_path.relative_to(REPOSITORY_ROOT))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in actual_owners:
                actual_owners[node.name].append(relative_path)
    assert actual_owners == {symbol: [owner] for symbol, owner in expected_owners.items()}
