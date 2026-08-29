"""Reference constructors for explicit legacy CPU-backend tests."""

from pyscf import dft, scf

from deepks.deephf.workflow import (
    _canonicalize_final_orbitals,
    _configure_gpu_dft,
)


REFERENCE_CLASSES = {
    "rhf": scf.hf.RHF,
    "uhf": scf.uhf.UHF,
    "rks": dft.rks.RKS,
    "uks": dft.uks.UKS,
}


def build_cpu_reference(molecule, family):
    """Build the exact CPU reference required by legacy backend unit tests."""
    reference = REFERENCE_CLASSES[family](molecule)
    reference.verbose = 0
    reference.conv_tol = 1.0e-12
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    if family in {"rks", "uks"}:
        _configure_gpu_dft(reference, molecule, None)
    reference.kernel()
    if not reference.converged:
        raise RuntimeError(f"the test {family.upper()} reference did not converge")
    _canonicalize_final_orbitals(reference)
    return reference
