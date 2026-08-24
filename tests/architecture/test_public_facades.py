import importlib

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


def test_facade_star_imports_match_the_deliberate_public_surface():
    allowed_dunders = {
        "__all__",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }
    for family in ("rhf", "uhf", "rks", "uks"):
        module = importlib.import_module(f"deepks.deephf.pyscf_{family}")
        namespace = {}
        exec(f"from deepks.deephf.pyscf_{family} import *", namespace)
        assert set(namespace) - {"__builtins__"} == set(module.__all__)
        assert set(vars(module)) - allowed_dunders == set(module.__all__)
        assert all(not name.startswith("_") for name in module.__all__)
