import numpy as np
from pyscf import gto

from deepks.deephf import build_reference, make_deephf
from deepks.gpu import as_numpy


def test_uks_reference_is_canonical_for_the_final_density():
    molecule = gto.M(
        atom="Li 0 0 0",
        basis="sto-3g",
        spin=1,
        verbose=0,
    )
    reference = build_reference(
        molecule,
        "uks",
        scf_args={"conv_tol": 1.0e-12, "conv_tol_grad": 1.0e-7},
    )
    coefficient = as_numpy(reference.mo_coeff)
    energy = as_numpy(reference.mo_energy)
    overlap = as_numpy(reference.get_ovlp())
    fock = as_numpy(reference.get_fock())
    residual = fock @ coefficient - overlap @ (coefficient * energy[:, None, :])
    residual_mo = np.einsum("smi,smj->sij", coefficient, residual)

    for spin_index in range(2):
        occupied = as_numpy(reference.mo_occ[spin_index]) > 0.0
        virtual = ~occupied
        np.testing.assert_allclose(
            residual_mo[spin_index][np.ix_(occupied, occupied)],
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            residual_mo[spin_index][np.ix_(virtual, virtual)],
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        )

    method = make_deephf(
        reference,
        None,
        projector_basis=[[0, [0.8, 1.0]]],
    )
    assert np.isfinite(method.kernel())
