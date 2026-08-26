import os
from dataclasses import dataclass

import numpy as np
import torch

from deepks.data.force_schema import (
    ForceDataError,
    validate_force_checkpoint_metadata,
)
from deepks.gpu import DEFAULT_CUDA_DEVICE, require_cuda_device
from deepks.model.model import CorrNet
from deepks.model.reader import (
    FORCE_MODE_DEEPHF_RELAXED,
    FORCE_MODE_NONE,
    GroupReader,
)
from deepks.model.train import Evaluator
from deepks.utils import check_list, load_dirs


DEVICE = torch.device(DEFAULT_CUDA_DEVICE)


@dataclass(frozen=True)
class SavedDataSystemResult:
    """Targets, predictions, and metrics for one saved-data system."""

    path: str
    energy_target: np.ndarray
    energy_prediction: np.ndarray
    energy_mae: float
    energy_rmse: float
    force_target: np.ndarray | None
    force_prediction: np.ndarray | None
    force_mae: float | None
    force_rmse: float | None


@dataclass(frozen=True)
class SavedDataTestResult:
    """Aggregate saved-data metrics with every per-system prediction."""

    energy_mae: float
    energy_rmse: float
    force_mae: float | None
    force_rmse: float | None
    systems: tuple[SavedDataSystemResult, ...]

    def __iter__(self):
        """Retain unpacking compatibility with the historical energy result."""
        yield self.energy_mae
        yield self.energy_rmse

    @property
    def energy_predictions(self) -> tuple[np.ndarray, ...]:
        return tuple(system.energy_prediction for system in self.systems)

    @property
    def force_predictions(self) -> tuple[np.ndarray, ...] | None:
        if self.force_mae is None:
            return None
        return tuple(system.force_prediction for system in self.systems)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    difference = prediction - target
    return (
        float(np.mean(np.abs(difference))),
        float(np.sqrt(np.mean(np.square(difference)))),
    )


def _validate_force_model(model, contract) -> None:
    if not isinstance(model, CorrNet):
        raise TypeError("strict force saved-data testing requires a CorrNet checkpoint")
    extra_info = getattr(model, "_checkpoint_extra_info", None)
    if not isinstance(extra_info, dict):
        raise ForceDataError("loaded model is missing checkpoint metadata")
    metadata = extra_info.get("force_training")
    validate_force_checkpoint_metadata(metadata, contract)


def _write_energy_result(filename, system: SavedDataSystemResult, header_prefix=""):
    header = (
        f"{header_prefix}energy MAE: {system.energy_mae}\n"
        f"energy RMSE: {system.energy_rmse}\n"
        "target_energy predicted_energy"
    )
    np.savetxt(
        filename,
        np.stack([system.energy_target, system.energy_prediction], axis=1),
        header=header,
    )


def _force_rows(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    frame, atom, cartesian = np.indices(target.shape)
    return np.column_stack(
        [
            frame.reshape(-1),
            atom.reshape(-1),
            cartesian.reshape(-1),
            target.reshape(-1),
            prediction.reshape(-1),
        ]
    )


def _write_force_result(filename, system: SavedDataSystemResult, header_prefix=""):
    header = (
        f"{header_prefix}force MAE: {system.force_mae}\n"
        f"force RMSE: {system.force_rmse}\n"
        "frame atom cartesian target_force predicted_force"
    )
    np.savetxt(
        filename,
        _force_rows(system.force_target, system.force_prediction),
        header=header,
        fmt=["%d", "%d", "%d", "%.18e", "%.18e"],
    )


def test(model, g_reader, dump_prefix="test", group=False, force_aware=None):
    """Test one loaded checkpoint against saved energy or strict relaxed-force data."""
    model.eval()
    contract = getattr(g_reader, "force_contract", None)
    contracts = getattr(
        g_reader,
        "force_contracts",
        (contract,) if contract is not None else (),
    )
    inferred_force_aware = contract is not None
    if force_aware is None:
        force_aware = inferred_force_aware
    if not isinstance(force_aware, bool):
        raise TypeError("force_aware must be bool or None")
    if force_aware and contract is None:
        raise ForceDataError(
            "force-aware saved-data testing requires a validated force-data contract"
        )
    if inferred_force_aware and not force_aware:
        raise ForceDataError(
            "a strict force dataset cannot be tested through the energy-only path"
        )
    if force_aware:
        _validate_force_model(model, contract)

    evaluator = Evaluator(
        energy_factor=1.0,
        force_factor=1.0 if force_aware else 0.0,
        force_contract=contracts if force_aware else None,
    )
    systems = []
    for index in range(g_reader.nsystems):
        sample = g_reader.sample_all(index)
        evaluation = evaluator.evaluate(model, sample, create_graph=False)
        energy_target = sample["energy"].detach().cpu().numpy().reshape(-1)
        energy_prediction = (
            evaluation.prediction.energy.detach().cpu().numpy().reshape(-1)
        )
        energy_mae, energy_rmse = _metrics(energy_target, energy_prediction)

        force_target = None
        force_prediction = None
        force_mae = None
        force_rmse = None
        if force_aware:
            force_target = sample["force"].detach().cpu().numpy()
            force_prediction = evaluation.prediction.force.detach().cpu().numpy()
            force_mae, force_rmse = _metrics(force_target, force_prediction)

        system = SavedDataSystemResult(
            path=g_reader.readers[index].data_path,
            energy_target=energy_target,
            energy_prediction=energy_prediction,
            energy_mae=energy_mae,
            energy_rmse=energy_rmse,
            force_target=force_target,
            force_prediction=force_prediction,
            force_mae=force_mae,
            force_rmse=force_rmse,
        )
        systems.append(system)
        if not group and dump_prefix is not None:
            digits = max(len(str(g_reader.nsystems)), 2)
            indexed_prefix = f"{dump_prefix}.{index:0{digits}d}"
            _write_energy_result(
                f"{indexed_prefix}.out",
                system,
                header_prefix=f"{system.path}\n",
            )
            if force_aware:
                _write_force_result(
                    f"{indexed_prefix}.force.out",
                    system,
                    header_prefix=f"{system.path}\n",
                )

    all_energy_target = np.concatenate(
        [system.energy_target for system in systems], axis=0
    )
    all_energy_prediction = np.concatenate(
        [system.energy_prediction for system in systems], axis=0
    )
    energy_mae, energy_rmse = _metrics(
        all_energy_target,
        all_energy_prediction,
    )
    force_mae = None
    force_rmse = None
    if force_aware:
        all_force_target = np.concatenate(
            [system.force_target.reshape(-1) for system in systems], axis=0
        )
        all_force_prediction = np.concatenate(
            [system.force_prediction.reshape(-1) for system in systems], axis=0
        )
        force_mae, force_rmse = _metrics(
            all_force_target,
            all_force_prediction,
        )

    result = SavedDataTestResult(
        energy_mae=energy_mae,
        energy_rmse=energy_rmse,
        force_mae=force_mae,
        force_rmse=force_rmse,
        systems=tuple(systems),
    )
    info = f"all systems energy MAE: {energy_mae}\nall systems energy RMSE: {energy_rmse}"
    if force_aware:
        info += f"\nall systems force MAE: {force_mae}\nall systems force RMSE: {force_rmse}"
    print(info)

    if dump_prefix is not None and group:
        aggregate_energy = SavedDataSystemResult(
            path="all systems",
            energy_target=all_energy_target,
            energy_prediction=all_energy_prediction,
            energy_mae=energy_mae,
            energy_rmse=energy_rmse,
            force_target=None,
            force_prediction=None,
            force_mae=None,
            force_rmse=None,
        )
        _write_energy_result(f"{dump_prefix}.out", aggregate_energy)
        if force_aware:
            force_rows = []
            for system_index, system in enumerate(systems):
                rows = _force_rows(system.force_target, system.force_prediction)
                force_rows.append(
                    np.column_stack(
                        [np.full(rows.shape[0], system_index), rows]
                    )
                )
            np.savetxt(
                f"{dump_prefix}.force.out",
                np.concatenate(force_rows, axis=0),
                header=(
                    f"force MAE: {force_mae}\n"
                    f"force RMSE: {force_rmse}\n"
                    "system frame atom cartesian target_force predicted_force"
                ),
                fmt=["%d", "%d", "%d", "%d", "%.18e", "%.18e"],
            )
    return result


def main(
    data_paths,
    model_file="model.pth",
    output_prefix="test",
    group=False,
    energy_name="e_corr_target",
    descriptor_name=("descriptor",),
    force_mode=FORCE_MODE_NONE,
):
    device = require_cuda_device(DEVICE)
    data_paths = load_dirs(data_paths)
    if isinstance(descriptor_name, (list, tuple)) and len(descriptor_name) == 1:
        descriptor_name = descriptor_name[0]
    if force_mode not in (FORCE_MODE_NONE, FORCE_MODE_DEEPHF_RELAXED):
        raise ValueError(
            "force_mode must be 'none' or 'deephf_relaxed' for saved-data testing"
        )
    g_reader = GroupReader(
        data_paths,
        energy_name=energy_name,
        descriptor_name=descriptor_name,
        converged_filter=False,
        extra_label=True,
        force_mode=force_mode,
    )
    contract = getattr(g_reader, "force_contract", None)
    model_files = check_list(model_file)
    results = []
    for filename in model_files:
        print(filename)
        directory = os.path.dirname(filename)
        model = CorrNet.load(
            filename,
            require_force_metadata=contract is not None,
            expected_force_contract_fingerprint=(
                contract.compatibility_fingerprint if contract is not None else None
            ),
            expected_force_contract=contract,
        ).double().to(device)
        dump = os.path.join(directory, output_prefix) if output_prefix is not None else None
        if dump is not None:
            output_directory = os.path.dirname(dump)
            if output_directory:
                os.makedirs(output_directory, exist_ok=True)
        if model.elem_table is not None:
            element_list, element_constants = model.elem_table
            g_reader.collect_elems(element_list)
            g_reader.subtract_elem_const(element_constants)
        results.append(
            test(
                model,
                g_reader,
                dump_prefix=dump,
                group=group,
                force_aware=contract is not None,
            )
        )
        g_reader.revert_elem_const()
    return results


if __name__ == "__main__":
    from deepks.main import test_cli as cli

    cli()
