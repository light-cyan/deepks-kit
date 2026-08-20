"""Perturbative DeePHF energy and strict molecular analytic gradients."""

from .adjoint import (
    AdjointDiagnostics,
    AdjointError,
    AdjointResult,
    ScalarAdjointProblem,
    solve_scalar_adjoint,
)

from .capabilities import (
    DeePHFCapabilityError,
    validate_force_model,
    validate_model,
    validate_model_output,
)
from .gradient import RHFDeePHFGradients
from .force_data import (
    RHFForceDataError,
    RHFForceFrame,
    generate_rhf_force_frame,
    write_rhf_force_dataset,
)
from .method import DeePHF
from .pyscf_rhf import (
    RHFAdjoint,
    RHFAdjointAdapter,
    RHFAdjointDiagnostics,
    RHFAdjointError,
    RHFResponse,
    RHFResponseAdapter,
    RHFResponseDiagnostics,
    RHFResponseError,
    RHFRootSnapshot,
    RHFScannerReferenceError,
    RHFScannerReferenceFactory,
    validate_reference,
)
from .scanner import RHFDeePHFGradientScanner, RHFDeePHFScannerError
from .uhf_gradient import UHFDeePHFGradients
from .uhf_method import UHFDeePHF
from .pyscf_uhf import (
    UHFResponse,
    UHFResponseAdapter,
    UHFResponseDiagnostics,
    UHFResponseError,
    validate_uhf_reference,
)
from .rks_gradient import RKSDeePHFGradients
from .rks_method import RKSDeePHF
from .pyscf_rks import (
    RKSFunctionalProvenance,
    RKSGridProvenance,
    RKSNativeGradient,
    RKSResponse,
    RKSResponseAdapter,
    RKSResponseDiagnostics,
    RKSResponseError,
    validate_rks_reference,
)
from .zvector import RHFDeePHFZVectorGradients

__all__ = [
    "DeePHF",
    "DeePHFCapabilityError",
    "AdjointDiagnostics",
    "AdjointError",
    "AdjointResult",
    "RHFDeePHFGradients",
    "RHFDeePHFGradientScanner",
    "RHFDeePHFScannerError",
    "RHFDeePHFZVectorGradients",
    "RHFForceDataError",
    "RHFForceFrame",
    "RHFAdjoint",
    "RHFAdjointAdapter",
    "RHFAdjointDiagnostics",
    "RHFAdjointError",
    "RHFResponse",
    "RHFResponseAdapter",
    "RHFResponseDiagnostics",
    "RHFResponseError",
    "RHFRootSnapshot",
    "RHFScannerReferenceError",
    "RHFScannerReferenceFactory",
    "RKSDeePHF",
    "RKSDeePHFGradients",
    "RKSFunctionalProvenance",
    "RKSGridProvenance",
    "RKSNativeGradient",
    "RKSResponse",
    "RKSResponseAdapter",
    "RKSResponseDiagnostics",
    "RKSResponseError",
    "ScalarAdjointProblem",
    "UHFDeePHF",
    "UHFDeePHFGradients",
    "UHFResponse",
    "UHFResponseAdapter",
    "UHFResponseDiagnostics",
    "UHFResponseError",
    "generate_rhf_force_frame",
    "solve_scalar_adjoint",
    "validate_model",
    "validate_force_model",
    "validate_model_output",
    "validate_reference",
    "validate_rks_reference",
    "validate_uhf_reference",
    "write_rhf_force_dataset",
]
