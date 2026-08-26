"""Shared deterministic inputs and result helpers for the validation campaign."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pyscf
import scipy
import torch
from pyscf import gto
from pyscf.dft import libxc

from deepks.deephf import build_reference, make_deephf
from deepks.model.model import CorrNet


VALIDATION_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = VALIDATION_DIR.parents[1]
CONFIG_PATH = VALIDATION_DIR / "configs" / "campaign.json"
GEOMETRY_DIR = VALIDATION_DIR / "geometries"
CHECKPOINT_DIR = VALIDATION_DIR / "checkpoints"
RUN_DIR = VALIDATION_DIR / "runs"
REPORT_DIR = VALIDATION_DIR / "reports"
AXES = ("x", "y", "z")
ANGSTROM_PER_BOHR = 0.529177210903


def load_config() -> dict[str, Any]:
    """Load the frozen campaign configuration."""
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def workload_by_id(workload_id: str) -> dict[str, Any]:
    """Return one frozen workload record."""
    for workload in load_config()["workloads"]:
        if workload["id"] == workload_id:
            return workload
    raise KeyError(f"unknown workload {workload_id!r}")


def configure_threads(thread_count: int) -> None:
    """Apply one consistent numerical-library thread budget."""
    if isinstance(thread_count, bool) or int(thread_count) <= 0:
        raise ValueError("thread_count must be a positive integer")
    value = str(int(thread_count))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value
    torch.set_num_threads(int(thread_count))
    torch.set_num_interop_threads(int(thread_count))


def read_xyz(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read one XYZ geometry without changing its Angstrom coordinates."""
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    records = [line.split() for line in lines[2 : 2 + atom_count]]
    if len(records) != atom_count or any(len(record) < 4 for record in records):
        raise ValueError(f"{path} does not contain {atom_count} atom records")
    atoms = tuple(record[0] for record in records)
    coordinates = np.asarray(
        [[float(value) for value in record[1:4]] for record in records],
        dtype=np.float64,
    )
    return atoms, coordinates


def workload_geometry(workload: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray]:
    """Return frozen atoms and coordinates in Bohr."""
    atoms, coordinates_angstrom = read_xyz(GEOMETRY_DIR / workload["geometry"])
    return atoms, coordinates_angstrom / ANGSTROM_PER_BOHR


def make_molecule(
    workload: dict[str, Any],
    coordinates_bohr: np.ndarray | None = None,
    atoms: tuple[str, ...] | None = None,
) -> gto.Mole:
    """Build one spherical, symmetry-disabled molecular input."""
    frozen_atoms, frozen_coordinates = workload_geometry(workload)
    if atoms is None:
        atoms = frozen_atoms
    if coordinates_bohr is None:
        coordinates_bohr = frozen_coordinates
    return gto.M(
        atom=list(zip(atoms, np.asarray(coordinates_bohr, dtype=np.float64))),
        basis=workload["basis"],
        unit="Bohr",
        charge=int(workload["charge"]),
        spin=int(workload["spin"]),
        symmetry=False,
        cart=False,
        verbose=0,
    )


def fresh_reference(
    workload: dict[str, Any],
    family: str,
    coordinates_bohr: np.ndarray | None = None,
    atoms: tuple[str, ...] | None = None,
):
    """Converge one fresh native reference under frozen controls."""
    molecule = make_molecule(workload, coordinates_bohr, atoms)
    return build_reference(
        molecule,
        family,
        scf_args=effective_scf_controls(workload, family),
        verbose=0,
    )


def _family_controls(workload: dict[str, Any], family: str) -> dict[str, Any]:
    """Return validated controls declared for one workload/reference family."""
    if family not in workload["families"]:
        raise ValueError(
            f"family {family!r} is not declared for workload {workload['id']!r}"
        )
    all_controls = workload.get("family_controls", {})
    unknown_families = set(all_controls).difference(workload["families"])
    if unknown_families:
        raise ValueError(
            f"workload {workload['id']!r} has controls for undeclared families: "
            f"{sorted(unknown_families)}"
        )
    controls = all_controls.get(family, {})
    unknown_keys = set(controls).difference(
        {"finite_difference_steps_bohr", "scf_overrides"}
    )
    if unknown_keys:
        raise ValueError(
            f"workload {workload['id']!r} family {family!r} has unknown controls: "
            f"{sorted(unknown_keys)}"
        )
    return controls


def effective_scf_controls(workload: dict[str, Any], family: str) -> dict[str, Any]:
    """Merge global, workload, and reference-family SCF controls."""
    controls = dict(load_config()["scf_controls"])
    controls.update(workload.get("scf_overrides", {}))
    controls.update(_family_controls(workload, family).get("scf_overrides", {}))
    return controls


def deterministic_model(scale: float | None = None) -> CorrNet:
    """Return the frozen nonlinear force-capable tanh CorrNet."""
    config = load_config()
    if scale is None:
        scale = float(config["model_scale"])
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(3,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=config["projector_basis"],
        input_shift=[0.17, -0.23, 0.11, -0.07],
        input_scale=[0.71, 1.29, 0.83, 1.17],
        output_scale=1.37,
    ).double()
    with torch.no_grad():
        model.linear.weight[:] = scale * torch.tensor(
            [[0.037, -0.021, 0.013, 0.029]], dtype=torch.float64
        )
        model.linear.bias.fill_(scale * 0.011)
        first_layer, output_layer = model.densenet.layers
        first_layer.weight[:] = torch.tensor(
            [
                [0.31, -0.22, 0.17, 0.09],
                [0.17, 0.29, -0.14, 0.23],
                [-0.26, 0.14, 0.28, -0.19],
            ],
            dtype=torch.float64,
        )
        first_layer.bias[:] = torch.tensor(
            [0.03, -0.04, 0.02], dtype=torch.float64
        )
        output_layer.weight[:] = scale * torch.tensor(
            [[0.27, -0.19, 0.16]], dtype=torch.float64
        )
        output_layer.bias.fill_(scale * 0.021)
        model.energy_const.fill_(scale * 0.019)
    return model.eval()


def make_method(reference, model=None):
    """Build the public DeePHF class matching a native reference."""
    config = load_config()
    method = make_deephf(
        reference,
        model,
        projector_basis=config["projector_basis"],
        device="cpu",
        response_options=config["response_controls"],
        adjoint_options=None,
    )
    allowed = getattr(method, "_zvector_options", None)
    if allowed is None:
        allowed = {
            "residual_tolerance",
            "orbital_gap_tolerance",
            "objective_symmetry_tolerance",
        }
        historical_operator_controls = {
            name: value
            for name, value in config["validation_operator_controls"].items()
            if name != "invariant_tolerance"
        }
        method.response_options.update(historical_operator_controls)
        allowed.update(historical_operator_controls)
    method.adjoint_options = {
        name: value
        for name, value in config["adjoint_controls"].items()
        if name in allowed
    }
    return method


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with SHA-256."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file with SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(value: np.ndarray) -> str:
    """Hash an array including dtype and shape."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    """Hash the frozen model architecture metadata and state tensors."""
    digest = hashlib.sha256()
    digest.update(
        f"{type(model).__module__}.{type(model).__qualname__}".encode("utf-8")
    )
    digest.update(
        json.dumps(
            json_safe(getattr(model, "_init_args", {})),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().numpy()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(repr(getattr(model, "_pbas", None)).encode("utf-8"))
    return digest.hexdigest()


def validation_hash() -> str:
    """Hash all compact validation inputs and executable scripts."""
    digest = hashlib.sha256()
    paths = [CONFIG_PATH]
    paths.extend(sorted(GEOMETRY_DIR.glob("*.xyz")))
    paths.extend(sorted((VALIDATION_DIR / "scripts").glob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(VALIDATION_DIR)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def deterministic_directions(workload: dict[str, Any]) -> np.ndarray:
    """Build five normalized full-coordinate directions from the frozen seed."""
    atom_count = len(workload_geometry(workload)[0])
    direction_count = int(load_config()["finite_difference"]["direction_count"])
    if direction_count != 5:
        raise ValueError("the scientific protocol requires exactly five directions")
    seed_material = f"{load_config()['seed']}:{workload['id']}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(direction_count, atom_count, 3))
    directions -= directions.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(directions.reshape(direction_count, -1), axis=1)
    directions /= norms[:, None, None]
    return directions


def finite_difference_components(workload: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    """Return the frozen Cartesian finite-difference component set."""
    atom_count = len(workload_geometry(workload)[0])
    if workload["finite_difference"] == "all":
        return tuple((atom, axis) for atom in range(atom_count) for axis in range(3))
    return tuple(tuple(int(value) for value in item) for item in workload.get("components", ()))


def finite_difference_steps(
    workload: dict[str, Any], family: str
) -> tuple[float, ...]:
    """Return the predeclared effective step set for one workload and family."""
    section = load_config()["finite_difference"]
    family_values = _family_controls(workload, family).get(
        "finite_difference_steps_bohr"
    )
    if family_values is not None:
        values = family_values
    elif workload["finite_difference"] == "all":
        values = section["small_steps_bohr"]
    elif workload["finite_difference"] == "selected":
        values = section["long_steps_bohr"]
    else:
        values = []
    steps = tuple(float(value) for value in values)
    if any(not math.isfinite(step) or step <= 0.0 for step in steps):
        raise ValueError("finite-difference steps must be finite and positive")
    if len(steps) != len(set(steps)):
        raise ValueError("finite-difference steps must be unique")
    return steps


def response_dimensions(reference) -> dict[str, Any]:
    """Summarize occupied, virtual, and response dimensions."""
    occupation = np.asarray(reference.mo_occ)
    if occupation.ndim == 1:
        occupied = int(np.count_nonzero(occupation > 0.0))
        virtual = int(np.count_nonzero(occupation == 0.0))
        return {
            "occupied_count": occupied,
            "virtual_count": virtual,
            "response_dimension": occupied * virtual,
        }
    occupied = tuple(int(np.count_nonzero(row > 0.0)) for row in occupation)
    virtual = tuple(int(np.count_nonzero(row == 0.0)) for row in occupation)
    return {
        "occupied_count": occupied,
        "virtual_count": virtual,
        "response_dimension": sum(o * v for o, v in zip(occupied, virtual, strict=True)),
    }


def minimum_orbital_gaps(reference) -> tuple[float, ...]:
    """Return occupied-virtual gaps for each spin channel."""
    occupation = np.asarray(reference.mo_occ)
    energy = np.asarray(reference.mo_energy)
    if occupation.ndim == 1:
        gap = np.min(energy[occupation == 0.0, None] - energy[occupation > 0.0])
        return (float(gap),)
    return tuple(
        float(
            np.min(
                energy[spin, occupation[spin] == 0.0, None]
                - energy[spin, occupation[spin] > 0.0]
            )
        )
        for spin in range(2)
    )


def occupied_subspace_overlap(reference, displaced) -> tuple[float, ...]:
    """Return minimum singular values of central/displaced occupied overlaps."""
    cross = np.asarray(gto.intor_cross("int1e_ovlp", reference.mol, displaced.mol))
    central_coefficients = np.asarray(reference.mo_coeff)
    displaced_coefficients = np.asarray(displaced.mo_coeff)
    central_occupations = np.asarray(reference.mo_occ)
    displaced_occupations = np.asarray(displaced.mo_occ)
    if central_occupations.ndim == 1:
        central_coefficients = central_coefficients[None]
        displaced_coefficients = displaced_coefficients[None]
        central_occupations = central_occupations[None]
        displaced_occupations = displaced_occupations[None]
    values = []
    for spin in range(central_occupations.shape[0]):
        central = central_coefficients[spin][:, central_occupations[spin] > 0.0]
        shifted = displaced_coefficients[spin][:, displaced_occupations[spin] > 0.0]
        singular_values = np.linalg.svd(central.T @ cross @ shifted, compute_uv=False)
        values.append(float(np.min(singular_values)))
    return tuple(values)


def state_continuity(reference, displaced) -> dict[str, Any]:
    """Audit the electronic state invariants for one displaced reference."""
    central_occupations = np.asarray(reference.mo_occ)
    displaced_occupations = np.asarray(displaced.mo_occ)
    checks = {
        "electron_count_equal": int(reference.mol.nelectron) == int(displaced.mol.nelectron),
        "spin_equal": int(reference.mol.spin) == int(displaced.mol.spin),
        "occupations_equal": bool(np.array_equal(central_occupations, displaced_occupations)),
        "ao_labels_equal": tuple(reference.mol.ao_labels()) == tuple(displaced.mol.ao_labels()),
        "occupied_subspace_minimum_overlap": occupied_subspace_overlap(reference, displaced),
        "minimum_orbital_gaps": minimum_orbital_gaps(displaced),
    }
    checks["accepted"] = bool(
        checks["electron_count_equal"]
        and checks["spin_equal"]
        and checks["occupations_equal"]
        and checks["ao_labels_equal"]
    )
    return checks


def gradient_partitions(driver) -> dict[str, Any]:
    """Collect available public gradient partitions without inventing fields."""
    names = (
        "reference_gradient",
        "correction_gradient_explicit",
        "correction_gradient_metric",
        "correction_gradient_adjoint_nuclear",
        "correction_gradient_adjoint_fixed_grid",
        "correction_gradient_adjoint_grid_coordinate",
        "correction_gradient_adjoint_grid_weight",
        "correction_gradient_adjoint_metric",
        "correction_gradient_grid_coordinate",
        "correction_gradient_grid_weight",
        "correction_gradient_occupied_virtual",
        "correction_gradient_response",
        "correction_gradient",
        "de_full",
        "de",
        "dq_dR_explicit",
        "dq_dR_response",
        "dq_dR_relaxed",
    )
    return {name: getattr(driver, name) for name in names if hasattr(driver, name)}


def diagnostics_dict(driver) -> dict[str, Any] | None:
    """Serialize common direct or adjoint diagnostics."""
    diagnostics = getattr(driver, "response_diagnostics", None)
    if diagnostics is None:
        return None
    return json_safe(diagnostics)


def max_abs(value: np.ndarray) -> float:
    """Return a finite maximum absolute value with an empty-array identity."""
    return float(np.max(np.abs(np.asarray(value)), initial=0.0))


def error_norms(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Return absolute and relative error norms."""
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    expected_norm = float(np.linalg.norm(expected))
    return {
        "max_abs": max_abs(difference),
        "l2": float(np.linalg.norm(difference)),
        "relative_l2": float(np.linalg.norm(difference) / max(expected_norm, np.finfo(float).tiny)),
    }


def statistics(samples: list[float]) -> dict[str, Any]:
    """Summarize raw timing samples with robust variability."""
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "samples": values,
        "median": median,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mad": mad,
        "mad_fraction": float(mad / median) if median else 0.0,
    }


def json_safe(value: Any) -> Any:
    """Convert scientific objects into strict JSON values."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict, stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        json_safe(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    temporary.replace(path)


def command_output(command: list[str], cwd: Path | None = None) -> str:
    """Return one best-effort command output for a run manifest."""
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def tracked_diff_sha256(source_root: Path) -> str:
    """Hash the complete tracked patch relative to the source base revision."""
    try:
        patch = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--"], cwd=source_root
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
    return sha256_bytes(patch)


def source_metadata() -> dict[str, Any]:
    """Record the exact source snapshot selected by the orchestrator."""
    source_root = Path(os.environ.get("DEEPKS_SOURCE_ROOT", REPOSITORY_DIR)).resolve()
    tracked_status = command_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], source_root
    )
    working_tree_status = command_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], source_root
    )
    tracked_status = os.environ.get(
        "VALIDATION_TRACKED_DIFF_STATUS", tracked_status
    )
    working_tree_status = os.environ.get(
        "VALIDATION_WORKING_TREE_STATUS", working_tree_status
    )
    base_revision = os.environ.get(
        "VALIDATION_BASE_REVISION",
        command_output(["git", "rev-parse", "HEAD"], source_root),
    )
    diff_hash = os.environ.get(
        "VALIDATION_TRACKED_DIFF_SHA256", tracked_diff_sha256(source_root)
    )
    source_clean = os.environ.get("VALIDATION_SOURCE_CLEAN")
    if source_clean is None:
        source_clean_value = working_tree_status == ""
    else:
        source_clean_value = source_clean == "1"
    return {
        "source_root": source_root,
        "base_revision": base_revision,
        "revision": base_revision,
        "source_clean": source_clean_value,
        "working_tree_status": working_tree_status,
        "tracked_diff_status": tracked_status,
        "tracked_source_clean": tracked_status == "",
        "tracked_diff_sha256": diff_hash,
        "config_hash": os.environ.get(
            "VALIDATION_CONFIG_HASH", sha256_file(CONFIG_PATH)
        ),
        "validation_hash": os.environ.get(
            "VALIDATION_INPUT_HASH", validation_hash()
        ),
        "source_uv_lock_hash": sha256_file(source_root / "uv.lock"),
        "execution_environment_uv_lock_hash": os.environ.get(
            "VALIDATION_ENV_LOCK_HASH"
        ),
        "python_prefix": sys.prefix,
    }


def environment_metadata(profile: str, threads: int) -> dict[str, Any]:
    """Collect software, hardware, affinity, and thread provenance."""
    numpy_config = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = numpy_config
        np.show_config()
    finally:
        sys.stdout = old_stdout
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    memory_total_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_total_kib = int(line.split()[1])
                break
    return {
        **source_metadata(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "pyscf": pyscf.__version__,
        "libxc": getattr(libxc, "__version__", None) or libxc.libxc_version(),
        "blas_configuration": numpy_config.getvalue(),
        "cpu_model": cpu_model,
        "physical_cores": os.cpu_count(),
        "logical_cores": os.cpu_count(),
        "memory_total_kib": memory_total_kib,
        "operating_system": platform.platform(),
        "process_affinity": sorted(os.sched_getaffinity(0)),
        "profile": profile,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "declared_threads": threads,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def base_result(
    action: str,
    profile: str,
    threads: int,
    workload: dict[str, Any] | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Create the common result envelope used by every child process."""
    result = {
        "schema_version": 1,
        "action": action,
        "profile": profile,
        "environment": environment_metadata(profile, threads),
        "categories": {
            "scientific": {"passed": None, "reasons": []},
            "integrity": {"passed": None, "reasons": []},
            "performance": {"passed": None, "reasons": []},
            "resource": {"passed": None, "reasons": []},
        },
    }
    if workload is not None:
        atoms, coordinates = workload_geometry(workload)
        result["case"] = workload["case"]
        result["workload_id"] = workload["id"]
        result["family"] = family
        result["charge"] = workload["charge"]
        result["spin"] = workload["spin"]
        result["basis"] = workload["basis"]
        result["atom_count"] = len(atoms)
        result["geometry_hash"] = hash_array(coordinates)
        result["geometry_file_hash"] = sha256_file(GEOMETRY_DIR / workload["geometry"])
        result["projector_hash"] = sha256_bytes(
            json.dumps(load_config()["projector_basis"], separators=(",", ":")).encode("utf-8")
        )
        result["model_hash"] = model_hash(deterministic_model())
        result["direction_hash"] = hash_array(deterministic_directions(workload))
        result["numerical_controls"] = {
            "scf": effective_scf_controls(workload, family),
            "finite_difference": {
                "mode": workload["finite_difference"],
                "steps_bohr": finite_difference_steps(workload, family),
                "components": finite_difference_components(workload),
                "direction_count": load_config()["finite_difference"][
                    "direction_count"
                ],
            },
            "response": load_config()["response_controls"],
            "adjoint": load_config()["adjoint_controls"],
            "dft": load_config()["dft"] if family in {"rks", "uks"} else None,
        }
    return result


def report_exception(result: dict[str, Any], category: str, error: BaseException) -> None:
    """Preserve a strict category failure and traceback text."""
    import traceback

    result["categories"][category]["passed"] = False
    result["categories"][category]["reasons"].append(
        f"{type(error).__name__}: {error}"
    )
    result["exception"] = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    diagnostic_context = getattr(error, "diagnostic_context", None)
    if diagnostic_context is not None:
        result["failure_context"] = json_safe(diagnostic_context)
    if error.__cause__ is not None:
        result["exception"]["cause"] = {
            "type": type(error.__cause__).__name__,
            "message": str(error.__cause__),
        }
