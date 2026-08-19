"""Run self-consistent DeePKS calculations and export canonical fields."""

import os
import sys
import time

import numpy as np
import torch
from pyscf import lib

from deepks.data.fields import required_field_labels, select_fields
from deepks.data.io import (
    build_molecule,
    collect_field_results,
    dump_data,
    dump_metadata,
    iter_system,
)
from deepks.model.model import CorrNet
from deepks.utils import (
    check_list,
    get_sys_name,
    load_sys_paths,
)

from .method import RDeePKS, UDeePKS
from .penalty import select_penalty


DEFAULT_FIELDS = {"converged", "descriptor", "e_base", "e_tot"}
DEFAULT_REFERENCE_ARGUMENTS = {"conv_tol": 1.0e-9}
DEFAULT_DEEPKS_ARGUMENTS = {"conv_tol": 1.0e-7}


def solve_molecule(
    mol,
    model,
    fields,
    labels=None,
    projector_basis=None,
    penalties=None,
    device=None,
    chkfile=None,
    verbose=0,
    **scf_arguments,
):
    """Run one DeePKS calculation and evaluate the selected fields."""
    start = time.time()
    method_class = RDeePKS if mol.spin == 0 else UDeePKS
    arguments = dict(scf_arguments)
    xc = arguments.pop("xc", "HF")
    grid_arguments = arguments.pop("grids", {})
    method = method_class(
        mol,
        model,
        xc=xc,
        projector_basis=projector_basis,
        penalties=penalties,
        device=device,
    )
    method.set(chkfile=chkfile, verbose=verbose)
    method.set(**arguments)
    method.grids.set(**grid_arguments)
    method.kernel()
    metadata = np.array(
        [
            method.n_descriptor_atoms,
            mol.natm,
            mol.nao,
            method.n_descriptor_features,
        ]
    )
    labels = {} if labels is None else labels
    results = {}
    for calculation_field in fields["scf"]:
        field_labels = {
            name: labels[name]
            for name in calculation_field.required_labels
        }
        results[calculation_field.name] = calculation_field.calculate(
            method,
            **field_labels,
        )
    if fields["gradient"]:
        gradient = method.nuc_grad_method().run()
        for calculation_field in fields["gradient"]:
            field_labels = {
                name: labels[name]
                for name in calculation_field.required_labels
            }
            results[calculation_field.name] = calculation_field.calculate(
                gradient,
                **field_labels,
            )
    if verbose:
        elapsed = time.time() - start
        print(
            f"DeePKS calculation time: {elapsed:6.2f}s, "
            f"converged: {method.converged}"
        )
    return metadata, results


def build_penalty(configuration, labels=None):
    labels = {} if labels is None else labels
    configuration = configuration.copy()
    penalty_type = configuration.pop("type")
    penalty_class = select_penalty(penalty_type)
    label_names = configuration.pop(
        "required_labels",
        penalty_class.required_labels,
    )
    label_arrays = [labels[name] for name in check_list(label_names)]
    return penalty_class(*label_arrays, **configuration)


def required_labels(fields, penalty_configurations=None):
    labels = required_field_labels(fields)
    for configuration in check_list(penalty_configurations):
        penalty_class = select_penalty(configuration["type"])
        labels.update(
            check_list(
                configuration.get(
                    "required_labels",
                    penalty_class.required_labels,
                )
            )
        )
    return labels


def main(
    systems,
    model_file="model.pth",
    basis="ccpvdz",
    projector_basis=None,
    penalty_terms=None,
    device=None,
    dump_dir=".",
    dump_fields=DEFAULT_FIELDS,
    group=False,
    mol_args=None,
    scf_args=None,
    verbose=0,
):
    if model_file is None or model_file.upper() == "NONE":
        model = None
        default_scf_arguments = DEFAULT_REFERENCE_ARGUMENTS
    else:
        model = CorrNet.load(model_file).double()
        default_scf_arguments = DEFAULT_DEEPKS_ARGUMENTS
    penalty_terms = check_list(penalty_terms)
    mol_args = {} if mol_args is None else mol_args
    scf_args = {} if scf_args is None else scf_args
    scf_arguments = {**default_scf_arguments, **scf_args}
    fields = select_fields(dump_fields)
    label_names = required_labels(
        fields["scf"] + fields["gradient"],
        penalty_terms,
    )
    if verbose:
        print(
            f"starting calculation with {lib.num_threads()} OpenMP threads "
            f"and {lib.param.MAX_MEMORY} MB maximum memory"
        )
        if verbose > 1:
            print(f"AO basis: {basis}")
            print(f"SCF arguments: {scf_arguments}")
    metadata = previous_metadata = None
    results = []
    systems = load_sys_paths(systems)
    for system_path in systems:
        system_path = system_path.rstrip(os.path.sep)
        for atoms, attributes, labels in iter_system(system_path, label_names):
            molecule_input = {
                **mol_args,
                "verbose": verbose,
                "atom": atoms,
                "basis": basis,
                **attributes,
            }
            mol = build_molecule(**molecule_input)
            penalties = [
                build_penalty(configuration, labels)
                for configuration in penalty_terms
            ]
            try:
                metadata, result = solve_molecule(
                    mol,
                    model,
                    fields,
                    labels,
                    projector_basis=projector_basis,
                    penalties=penalties,
                    device=device,
                    verbose=verbose,
                    **scf_arguments,
                )
            except Exception as error:
                print(
                    f"{system_path} failed: {error}",
                    file=sys.stderr,
                )
                raise
            if (
                group
                and previous_metadata is not None
                and np.any(metadata != previous_metadata)
            ):
                break
            results.append(result)
        if not group:
            output_directory = os.path.join(
                dump_dir,
                get_sys_name(os.path.basename(system_path)),
            )
            dump_metadata(output_directory, metadata)
            dump_data(
                output_directory,
                **collect_field_results(fields, metadata, results),
            )
            results = []
        elif (
            previous_metadata is not None
            and np.any(metadata != previous_metadata)
        ):
            print(
                f"{system_path} metadata does not match the grouped data",
                file=sys.stderr,
            )
            break
        previous_metadata = metadata
        if verbose:
            print(f"{system_path} finished")
    if group:
        dump_metadata(dump_dir, metadata)
        dump_data(
            dump_dir,
            **collect_field_results(fields, metadata, results),
        )
        if verbose:
            print("group finished")


if __name__ == "__main__":
    from deepks.main import scf_cli

    scf_cli()
