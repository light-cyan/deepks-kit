"""Internal implementation extracted from pyscf_rks.py."""

import numpy as np
from pyscf.grad import rks as rks_grad
from .pyscf_dft_provenance import RKSResponseError, _validated_float64_array, _validate_dft_implementations
from .pyscf_rks_reference import validate_rks_reference, rks_reference_fingerprint

def native_rks_gradient(reference, atom_indices=None) -> np.ndarray:
    """Evaluate one selected native RKS gradient with grid response."""
    validate_rks_reference(reference)
    _validate_dft_implementations("RKS")
    from .driver import validate_atom_indices

    selected = validate_atom_indices(reference.mol, atom_indices)
    atom_indices = tuple(range(reference.mol.natm)) if selected is None else selected
    initial_fingerprint = rks_reference_fingerprint(reference)
    try:
        driver = rks_grad.Gradients(reference)
        if type(driver) is not rks_grad.Gradients:
            raise RKSResponseError("the native RKS gradient driver type is invalid")
        driver.grids = reference.grids
        driver.grid_response = True
        gradient = driver.kernel(atmlst=list(atom_indices))
    except RKSResponseError:
        raise
    except Exception as error:
        raise RKSResponseError(f"PySCF native RKS gradient failed: {error}") from error
    gradient = _validated_float64_array(
        gradient,
        (len(atom_indices), 3),
        "native RKS gradient",
    )
    _validate_dft_implementations("RKS")
    validate_rks_reference(reference)
    if rks_reference_fingerprint(reference) != initial_fingerprint:
        raise RKSResponseError("the RKS reference changed during native gradient evaluation")
    return gradient
