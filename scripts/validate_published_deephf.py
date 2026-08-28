#!/usr/bin/env python
"""Validate the published DeePHF model, baseline, and analytic gradient on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deepks.data.io import build_molecule
from deepks.deephf import build_reference, make_deephf
from deepks.gpu import as_numpy, require_cuda_device
from deepks.model.model import CorrNet


def molecule_from_atom(atom: np.ndarray):
    return build_molecule(
        atom=[(int(row[0]), row[1:]) for row in atom],
        basis="def2-tzvp",
        unit="Bohr",
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    parser.add_argument("--system", default="gram_01_rxn000026_p000026_0")
    parser.add_argument("--grid-mode", choices=("default", "strict"), default="default")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--small-rho-cutoff", type=float, default=0.0)
    parser.add_argument("--finite-difference-step", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = require_cuda_device()
    model = CorrNet.load(args.assets / "b3lyp_gram_t1x.pth", strict=True).double().to(device).eval()
    archival_errors = {}
    with torch.no_grad():
        for path in sorted((args.assets / "reference").glob("*.npz")):
            data = np.load(path)
            descriptor = torch.as_tensor(data["dm_eig"], dtype=torch.float64, device=device)
            charges = np.rint(data["atom"][0, :, 0]).astype(int)
            predicted = float(model(descriptor).item() + model.get_elem_const(charges))
            target = float(data["l_e_delta"].reshape(-1)[0])
            archival_errors[path.stem] = {
                "error_hartree": predicted - target,
                "predicted_correction_hartree": predicted,
                "target_correction_hartree": target,
            }
    selected = np.load(args.assets / "reference" / f"{args.system}.npz")
    atom = np.asarray(selected["atom"][0], dtype=np.float64)
    dft_args = {
        "xc": "B3LYP5",
        "grid_mode": args.grid_mode,
        "grid_level": args.grid_level,
        "small_rho_cutoff": args.small_rho_cutoff,
    }
    scf_args = {
        "conv_tol": 1.0e-11,
        "conv_tol_grad": 1.0e-8,
        "conv_tol_cpscf": 1.0e-11,
        "max_cycle": 100,
        "newton_max_cycle": 50,
        "diis_space": 12,
    }
    reference = build_reference(
        molecule_from_atom(atom),
        "rks",
        scf_args=scf_args,
        dft_args=dft_args,
    )
    method = make_deephf(reference, model)
    energy = float(method.kernel())
    descriptor = method.descriptor()
    gradient = method.gradient()
    step = args.finite_difference_step
    displaced_energies = []
    initial_density = reference.make_rdm1().copy()
    for sign in (-1.0, 1.0):
        displaced = atom.copy()
        displaced[0, 1] += sign * step
        displaced_reference = build_reference(
            molecule_from_atom(displaced),
            "rks",
            scf_args=scf_args,
            dft_args=dft_args,
            dm0=initial_density,
        )
        displaced_energies.append(
            float(make_deephf(displaced_reference, model).kernel())
        )
    numerical = (displaced_energies[1] - displaced_energies[0]) / (2.0 * step)
    analytic = float(gradient[0, 0])
    stored_descriptor = np.asarray(selected["dm_eig"][0], dtype=np.float64)
    stored_base = float(selected["e_base"].reshape(-1)[0])
    maximum_archival_error = max(
        abs(values["error_hartree"]) for values in archival_errors.values()
    )
    baseline_error = float(reference.e_tot) - stored_base
    descriptor_error = float(np.max(np.abs(descriptor - stored_descriptor)))
    gradient_error = analytic - numerical
    checks = {
        "analytic_gradient_error_below_1e-5": abs(gradient_error) < 1.0e-5,
        "archival_correction_error_below_5e-4": maximum_archival_error < 5.0e-4,
        "baseline_error_below_1e-6": abs(baseline_error) < 1.0e-6,
        "descriptor_error_below_1e-5": descriptor_error < 1.0e-5,
    }
    result = {
        "analytic_gradient_hartree_per_bohr": analytic,
        "analytic_minus_finite_difference_hartree_per_bohr": gradient_error,
        "archival_correction_errors": archival_errors,
        "baseline": {
            "computed_hartree": float(reference.e_tot),
            "computed_minus_stored_hartree": baseline_error,
            "stored_hartree": stored_base,
        },
        "checks": checks,
        "correction_hartree": float(method.e_corr),
        "descriptor_maximum_absolute_error": descriptor_error,
        "finite_difference_gradient_hartree_per_bohr": numerical,
        "finite_difference_step_bohr": step,
        "gpu": torch.cuda.get_device_name(device),
        "reference": {
            "basis": "def2-tzvp",
            "dft_args": reference._deepks_dft_args,
            "family": "RKS",
            "xc": reference.xc,
        },
        "system": args.system,
        "total_energy_hartree": energy,
    }
    payload = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not all(checks.values()):
        raise RuntimeError("published DeePHF validation did not satisfy every check")


if __name__ == "__main__":
    main()
