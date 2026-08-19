import numpy as np
import torch
from pyscf import gto

from deepks.model.model import CorrNet
from deepks.scf.scf import DSCF
from deepks.utils import DEFAULT_BASIS, get_shell_sec


def test_restricted_scf_and_analytic_gradient_smoke():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    model = CorrNet(
        input_dim=sum(get_shell_sec(DEFAULT_BASIS)), hidden_sizes=(4,)
    ).double()
    # A zero correction recovers the underlying HF result while exercising the
    # projected-density and analytic-gradient integration paths.
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
        for parameter in model.densenet.parameters():
            parameter.zero_()

    mf = DSCF(mol, model)
    mf.conv_tol = 1e-10
    energy = mf.kernel()
    gradient = mf.nuc_grad_method().kernel()

    assert mf.converged
    assert np.isfinite(energy)
    assert gradient.shape == (2, 3)
    assert np.isfinite(gradient).all()
