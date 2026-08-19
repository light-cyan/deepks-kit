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


def _imported_modules(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    package_parts = _package_path(source_path)
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
                imported.append(".".join(target_parts))
            elif node.module:
                imported.append(node.module)
    return imported


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
