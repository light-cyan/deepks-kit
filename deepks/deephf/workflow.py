"""Public construction and inference workflow for strict DeePHF methods."""

import os

import numpy as np
from pyscf import dft, scf

from deepks.data.io import build_molecule, dump_data, iter_system
from deepks.model.model import CorrNet
from deepks.utils import get_sys_name, load_sys_paths

from .method import DeePHF
from .rks_method import RKSDeePHF
from .uhf_method import UHFDeePHF
from .uks_method import UKSDeePHF


METHOD_CLASSES = {
    scf.hf.RHF: DeePHF,
    scf.uhf.UHF: UHFDeePHF,
    dft.rks.RKS: RKSDeePHF,
    dft.uks.UKS: UKSDeePHF,
}
REFERENCE_FAMILIES = frozenset(("rhf", "uhf", "rks", "uks"))


def make_deephf(reference, model, *, projector_basis=None, device="cpu", response_options=None, adjoint_options=None):
    """Construct the exact public DeePHF method matching a native reference."""
    method_class = METHOD_CLASSES.get(type(reference))
    if method_class is None:
        raise TypeError("the reference type has no strict DeePHF method")
    if projector_basis is None and model is not None:
        projector_basis = model._pbas
    return method_class(
        reference,
        model,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )


def build_reference(molecule, family, *, scf_args=None, verbose=0):
    """Build and converge one exact native reference in the accepted support tier."""
    if type(family) is not str or family.lower() not in REFERENCE_FAMILIES:
        raise ValueError("reference family must be one of rhf, uhf, rks, or uks")
    family = family.lower()
    if family in {"rhf", "rks"} and molecule.spin != 0:
        raise ValueError(f"the strict {family.upper()} family requires molecular spin zero")
    if family == "rhf":
        reference = scf.RHF(molecule)
    elif family == "uhf":
        reference = scf.UHF(molecule)
    elif family == "rks":
        reference = dft.RKS(molecule)
    else:
        reference = dft.UKS(molecule)
    expected_type = {"rhf": scf.hf.RHF, "uhf": scf.uhf.UHF, "rks": dft.rks.RKS, "uks": dft.uks.UKS}[family]
    if type(reference) is not expected_type:
        raise TypeError(f"the requested {family.upper()} constructor did not produce its exact native reference type")
    reference.verbose = verbose
    controls = {
        "conv_tol": 1.0e-13,
        "conv_tol_grad": 1.0e-10,
        "conv_tol_cpscf": 1.0e-12,
        "max_cycle": 100,
    }
    controls.update({} if scf_args is None else dict(scf_args))
    unknown = sorted(set(controls) - {"conv_tol", "conv_tol_grad", "conv_tol_cpscf", "max_cycle", "diis_space", "level_shift", "direct_scf", "conv_check"})
    if unknown:
        raise ValueError("unsupported strict reference controls: " + ", ".join(unknown))
    reference.set(**controls)
    if family in {"rks", "uks"}:
        reference.xc = "LDA_X + LDA_C_VWN"
        reference.grids.atom_grid = {symbol: (20, 50) for symbol in set(molecule.elements)}
        reference.grids.prune = None
        reference.grids.alignment = 1
        reference.grids.cutoff = 1.0e-15
        reference.grids.build(with_non0tab=True, sort_grids=False)
        reference.small_rho_cutoff = 0.0
    reference.kernel(dm0=None)
    if not reference.converged:
        raise RuntimeError(f"the native {family.upper()} reference did not converge")
    return reference


def evaluate_molecule(molecule, model, *, family, backend="direct", projector_basis=None, device="cpu", scf_args=None, response_options=None, adjoint_options=None, verbose=0):
    """Evaluate energy, descriptor, gradient, and force through one public backend."""
    reference = build_reference(molecule, family, scf_args=scf_args, verbose=verbose)
    method = make_deephf(
        reference,
        model,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )
    energy = float(method.kernel())
    gradient_driver = method.nuc_grad_method(backend=backend)
    gradient = np.asarray(gradient_driver.kernel())
    return {
        "converged": np.asarray(True),
        "e_base": np.asarray(reference.e_tot, dtype=np.float64),
        "e_corr": np.asarray(energy - reference.e_tot, dtype=np.float64),
        "e_tot": np.asarray(energy, dtype=np.float64),
        "descriptor": np.asarray(method.descriptor(), dtype=np.float64),
        "gradient": gradient,
        "force": -gradient,
    }


def main(
    systems,
    *,
    reference,
    model_file=None,
    basis="sto-3g",
    projector_basis=None,
    backend="direct",
    device="cpu",
    dump_dir=".",
    mol_args=None,
    scf_args=None,
    response_options=None,
    adjoint_options=None,
    verbose=0,
):
    """Run strict DeePHF inference for every frame and persist canonical outputs."""
    if model_file is None or str(model_file).upper() == "NONE":
        model = None
    else:
        model = CorrNet.load(model_file, strict=True).double().eval()
    mol_args = {} if mol_args is None else dict(mol_args)
    outputs = []
    for system_path in load_sys_paths(systems):
        frames = []
        for atoms, attributes, _labels in iter_system(system_path):
            molecule_input = {
                **mol_args,
                "atom": atoms,
                "basis": basis,
                "verbose": verbose,
                **attributes,
            }
            molecule = build_molecule(**molecule_input)
            frames.append(
                evaluate_molecule(
                    molecule,
                    model,
                    family=reference,
                    backend=backend,
                    projector_basis=projector_basis,
                    device=device,
                    scf_args=scf_args,
                    response_options=response_options,
                    adjoint_options=adjoint_options,
                    verbose=verbose,
                )
            )
        if not frames:
            raise ValueError(f"system {system_path!r} contains no molecular frames")
        collected = {name: np.asarray([frame[name] for frame in frames]) for name in frames[0]}
        output_directory = os.path.join(dump_dir, get_sys_name(system_path))
        dump_data(output_directory, **collected)
        outputs.append((output_directory, collected))
    return outputs


__all__ = ["build_reference", "evaluate_molecule", "main", "make_deephf"]
