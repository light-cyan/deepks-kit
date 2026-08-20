"""Shared deterministic inputs and reporting helpers for release validation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
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
import torch
from pyscf import gto

from deepks.deephf import build_reference, make_deephf
from deepks.model.model import CorrNet


VALIDATION_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = VALIDATION_DIR.parent
CONFIG_PATH = VALIDATION_DIR / "configs" / "validation.json"
OUTPUT_DIR = VALIDATION_DIR / "outputs"
REPORT_DIR = VALIDATION_DIR / "reports"
PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]
AXIS_NAMES = ("x", "y", "z")


def configure_single_thread() -> None:
    """Pin numerical libraries before expensive work starts."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_xyz(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    records = [line.split() for line in lines[2 : 2 + count]]
    if len(records) != count:
        raise ValueError(f"{path} does not contain {count} atom records")
    atoms = tuple(record[0] for record in records)
    coordinates = np.asarray(
        [[float(value) for value in record[1:4]] for record in records],
        dtype=np.float64,
    )
    return atoms, coordinates


def molecule(
    atoms: tuple[str, ...],
    coordinates: np.ndarray,
    *,
    unit: str,
    spin: int,
    basis: str | None = None,
) -> gto.Mole:
    config = load_config()
    return gto.M(
        atom=list(zip(atoms, np.asarray(coordinates, dtype=np.float64))),
        basis=config["basis"] if basis is None else basis,
        unit=unit,
        spin=spin,
        charge=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def fresh_reference(
    family: str,
    atoms: tuple[str, ...],
    coordinates_bohr: np.ndarray,
    *,
    basis: str | None = None,
):
    config = load_config()
    spin = 0 if family in {"rhf", "rks"} else 1
    mol = molecule(
        atoms,
        coordinates_bohr,
        unit="Bohr",
        spin=spin,
        basis=basis,
    )
    return build_reference(
        mol,
        family,
        scf_args=config["scf_controls"],
        verbose=0,
    )


def deterministic_model(*, scale: float = 0.2) -> CorrNet:
    """Return the frozen nonlinear tanh CorrNet shared by all science gates."""
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(3,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=PROJECTOR_BASIS,
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


def zero_trainable_model() -> CorrNet:
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(8, 8),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        torch.manual_seed(int(load_config()["seed"]))
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
        for layer in model.densenet.layers[:-1]:
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.2)
        torch.nn.init.xavier_uniform_(model.densenet.layers[-1].weight, gain=0.02)
    return model


def make_method(reference, model):
    return make_deephf(
        reference,
        model,
        projector_basis=PROJECTOR_BASIS,
        device="cpu",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
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


def error_statistics(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atom_axis: int = 0,
    coordinate_axis: int = 1,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    error = actual - expected
    flat_index = int(np.argmax(np.abs(error)))
    index = tuple(int(item) for item in np.unravel_index(flat_index, error.shape))
    result = {
        "max_abs": float(np.max(np.abs(error), initial=0.0)),
        "rms": float(np.sqrt(np.mean(np.square(error)))),
        "worst_index": list(index),
        "actual_at_worst": float(actual[index]),
        "expected_at_worst": float(expected[index]),
        "signed_error_at_worst": float(error[index]),
    }
    if len(index) > max(atom_axis, coordinate_axis):
        result["worst_atom"] = index[atom_axis]
        result["worst_coordinate"] = AXIS_NAMES[index[coordinate_axis]]
    return result


def finite_difference(
    plus: dict[tuple[float, int, int], np.ndarray],
    minus: dict[tuple[float, int, int], np.ndarray],
    *,
    step: float,
    atom_count: int,
) -> np.ndarray:
    sample = np.asarray(plus[(step, 0, 0)])
    result = np.empty((atom_count, 3, *sample.shape), dtype=np.float64)
    for atom in range(atom_count):
        for coordinate in range(3):
            result[atom, coordinate] = (
                np.asarray(plus[(step, atom, coordinate)])
                - np.asarray(minus[(step, atom, coordinate)])
            ) / (2.0 * step)
    return result


def state_summary(reference) -> dict[str, Any]:
    occupations = np.asarray(reference.mo_occ, dtype=np.float64)
    energies = np.asarray(reference.mo_energy, dtype=np.float64)
    if occupations.ndim == 1:
        occupied = occupations > 0.0
        gaps = energies[~occupied, None] - energies[occupied]
        electron_counts: float | list[float] = float(np.sum(occupations))
        dimensions: int | list[int] = int(np.count_nonzero(~occupied) * np.count_nonzero(occupied))
        minimum_gap: float | list[float] = float(np.min(gaps))
    else:
        electron_counts = [float(np.sum(spin)) for spin in occupations]
        dimensions = []
        minimum_gap = []
        for spin in range(2):
            occupied = occupations[spin] > 0.0
            gaps = energies[spin][~occupied, None] - energies[spin][occupied]
            dimensions.append(int(np.count_nonzero(~occupied) * np.count_nonzero(occupied)))
            minimum_gap.append(float(np.min(gaps)))
    return {
        "converged": bool(reference.converged),
        "electron_counts": electron_counts,
        "occupations": occupations,
        "minimum_orbital_gap": minimum_gap,
        "response_dimensions": dimensions,
        "spin": int(reference.mol.spin),
        "charge": int(reference.mol.charge),
        "ao_count": int(reference.mol.nao),
    }


def occupied_subspace_overlap(reference, displaced) -> list[float]:
    cross_overlap = np.asarray(
        gto.intor_cross("int1e_ovlp", reference.mol, displaced.mol),
        dtype=np.float64,
    )
    central_occ = np.asarray(reference.mo_occ)
    displaced_occ = np.asarray(displaced.mo_occ)
    central_coeff = np.asarray(reference.mo_coeff)
    displaced_coeff = np.asarray(displaced.mo_coeff)
    if central_occ.ndim == 1:
        central_occ = central_occ[None, :]
        displaced_occ = displaced_occ[None, :]
        central_coeff = central_coeff[None, :, :]
        displaced_coeff = displaced_coeff[None, :, :]
    values = []
    for spin in range(central_occ.shape[0]):
        left = central_coeff[spin][:, central_occ[spin] > 0.0]
        right = displaced_coeff[spin][:, displaced_occ[spin] > 0.0]
        overlap = left.T @ cross_overlap @ right
        values.append(float(np.min(np.linalg.svd(overlap, compute_uv=False))))
    return values


def environment_metadata() -> dict[str, Any]:
    try:
        libxc_version = pyscf.dft.libxc.__version__
    except AttributeError:
        libxc_version = pyscf.dft.libxc.libxc_version()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cpu_model = "unknown"
    physical_cores = None
    logical_cores = os.cpu_count()
    try:
        lscpu = subprocess.run(
            ["lscpu", "-J"], check=True, capture_output=True, text=True
        )
        fields = {
            item["field"].rstrip(":"): item["data"]
            for item in json.loads(lscpu.stdout)["lscpu"]
        }
        cpu_model = fields.get("Model name", cpu_model)
        sockets = int(fields.get("Socket(s)", "1"))
        cores_per_socket = int(fields.get("Core(s) per socket", logical_cores or 1))
        physical_cores = sockets * cores_per_socket
    except (FileNotFoundError, ValueError, KeyError, subprocess.SubprocessError):
        pass
    memory_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "git_revision": git,
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pyscf": pyscf.__version__,
        "libxc": libxc_version,
        "cpu_model": cpu_model,
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "memory_bytes": memory_bytes,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "configuration_sha256": sha256_file(CONFIG_PATH),
    }


def command_record(command: list[str], *, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        command,
        cwd=REPOSITORY_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(process.stdout, encoding="utf-8")
    return {
        "command": command,
        "return_code": process.returncode,
        "log": str(log_path.relative_to(REPOSITORY_DIR)),
        "passed": process.returncode == 0,
    }


def report_exception(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "stage": stage,
        "passed": False,
        "exception_type": type(error).__name__,
        "message": str(error),
    }


def ensure_runtime() -> None:
    if sys.version_info[:2] < (3, 10):
        raise RuntimeError("validation requires Python 3.10 or newer")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)

