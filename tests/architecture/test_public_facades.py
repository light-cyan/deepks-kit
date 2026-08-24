from deepks.deephf.pyscf_rhf import (
    RHFAdjointAdapter,
    RHFResponseAdapter,
    RHFScannerReferenceFactory,
    validate_reference,
)
from deepks.deephf.pyscf_rks import (
    RKSAdjointAdapter,
    RKSResponseAdapter,
    native_rks_gradient,
    validate_rks_reference,
)
from deepks.deephf.pyscf_uhf import (
    UHFAdjointAdapter,
    UHFResponseAdapter,
    validate_uhf_reference,
)
from deepks.deephf.pyscf_uks import (
    UKSAdjointAdapter,
    UKSResponseAdapter,
    native_uks_gradient,
    validate_uks_reference,
)


def test_compatibility_facades_publish_supported_entry_points():
    symbols = (
        RHFAdjointAdapter,
        RHFResponseAdapter,
        RHFScannerReferenceFactory,
        validate_reference,
        RKSAdjointAdapter,
        RKSResponseAdapter,
        native_rks_gradient,
        validate_rks_reference,
        UHFAdjointAdapter,
        UHFResponseAdapter,
        validate_uhf_reference,
        UKSAdjointAdapter,
        UKSResponseAdapter,
        native_uks_gradient,
        validate_uks_reference,
    )
    assert all(callable(symbol) for symbol in symbols)


def test_facades_do_not_own_solver_implementations():
    modules = {
        symbol.__module__
        for symbol in (
            RHFAdjointAdapter,
            RHFResponseAdapter,
            RKSAdjointAdapter,
            RKSResponseAdapter,
            UHFAdjointAdapter,
            UHFResponseAdapter,
            UKSAdjointAdapter,
            UKSResponseAdapter,
        )
    }
    assert all(not name.endswith(("pyscf_rhf", "pyscf_uhf", "pyscf_rks", "pyscf_uks")) for name in modules)
