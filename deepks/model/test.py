import os
import numpy as np
import torch
import torch.nn as nn
try:
    import deepks
except ImportError as e:
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.model.model import CorrNet
from deepks.model.reader import GroupReader
from deepks.utils import load_yaml, load_dirs, check_list


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, g_reader, dump_prefix="test", group=False):
    model.eval()
    loss_fn = nn.MSELoss()
    energy_list = []
    predicted_energy_list = []

    for i in range(g_reader.nsystems):
        sample = g_reader.sample_all(i)
        nframes = sample["energy"].shape[0]
        sample = {k: v.to(DEVICE, non_blocking=True) for k, v in sample.items()}
        energy = sample["energy"]
        descriptor = sample["descriptor"]
        predicted_energy = model(descriptor)
        error = torch.sqrt(loss_fn(predicted_energy, energy))

        error_np = error.item()
        energy_np = energy.cpu().numpy().reshape(nframes, -1).sum(axis=1)
        predicted_energy_np = (
            predicted_energy.detach().cpu().numpy().reshape(nframes, -1).sum(axis=1)
        )
        error_l1 = np.mean(np.abs(energy_np - predicted_energy_np))
        energy_list.append(energy_np)
        predicted_energy_list.append(predicted_energy_np)

        if not group and dump_prefix is not None:
            nd = max(len(str(g_reader.nsystems)), 2)
            dump_res = np.stack([energy_np, predicted_energy_np], axis=1)
            header = f"{g_reader.path_list[i]}\nmean l1 error: {error_l1}\nmean l2 error: {error_np}\nenergy  predicted_energy"
            filename = f"{dump_prefix}.{i:0{nd}}.out"
            np.savetxt(filename, dump_res, header=header)
            # print(f"system {i} finished")

    all_energy = np.concatenate(energy_list, axis=0)
    all_predicted_energy = np.concatenate(predicted_energy_list, axis=0)
    all_err_l1 = np.mean(np.abs(all_energy - all_predicted_energy))
    all_err_l2 = np.sqrt(np.mean((all_energy - all_predicted_energy) ** 2))
    info = f"all systems mean l1 error: {all_err_l1}\nall systems mean l2 error: {all_err_l2}"
    print(info)
    if dump_prefix is not None and group:
        np.savetxt(
            f"{dump_prefix}.out",
            np.stack([all_energy, all_predicted_energy], axis=1),
            header=info + "\nenergy  predicted_energy",
        )
    return all_err_l1, all_err_l2


def main(data_paths, model_file="model.pth", 
         output_prefix='test', group=False,
         energy_name='e_corr_target', descriptor_name=('descriptor',)):
    data_paths = load_dirs(data_paths)
    if isinstance(descriptor_name, (list, tuple)) and len(descriptor_name) == 1:
        descriptor_name = descriptor_name[0]
    g_reader = GroupReader(
        data_paths,
        energy_name=energy_name,
        descriptor_name=descriptor_name,
        converged_filter=False,
        extra_label=True,
    )
    model_file = check_list(model_file)
    for f in model_file:
        print(f)
        p = os.path.dirname(f)
        model = CorrNet.load(f).double().to(DEVICE)
        dump = os.path.join(p, output_prefix)
        dir_name = os.path.dirname(dump)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if model.elem_table is not None:
            element_list, element_constants = model.elem_table
            g_reader.collect_elems(element_list)
            g_reader.subtract_elem_const(element_constants)
        test(model, g_reader, dump_prefix=dump, group=group)
        g_reader.revert_elem_const()


if __name__ == "__main__":
    from deepks.main import test_cli as cli
    cli()
