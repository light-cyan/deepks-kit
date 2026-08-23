"""Perturbative DeePHF energy and strict molecular analytic gradients."""

from .adjoint import (
    AdjointDiagnostics,
    AdjointError,
    AdjointResult,
    ScalarAdjointProblem,
    scalar_operator_fingerprint,
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
    RHFBlockedResponseSummary,
    RHFResponseDiagnostics,
    RHFResponseError,
    RHFRootSnapshot,
    RHFScannerReferenceError,
    RHFScannerReferenceFactory,
    validate_reference,
)
from .scanner import RHFDeePHFGradientScanner, RHFDeePHFScannerError
from .uhf_gradient import UHFDeePHFGradients
from .uhf_zvector import UHFDeePHFZVectorGradients
from .uhf_method import UHFDeePHF
from .pyscf_uhf import (
    UHFAdjoint,
    UHFAdjointAdapter,
    UHFAdjointDiagnostics,
    UHFAdjointError,
    UHFResponse,
    UHFResponseAdapter,
    UHFResponseDiagnostics,
    UHFResponseError,
    validate_uhf_reference,
)
from .rks_gradient import RKSDeePHFGradients
from .rks_method import RKSDeePHF
from .pyscf_rks import (
    RKSAdjoint,
    RKSAdjointAdapter,
    RKSAdjointDiagnostics,
    RKSAdjointError,
    RKSFunctionalProvenance,
    RKSGridProvenance,
    RKSNativeGradient,
    RKSResponse,
    RKSResponseAdapter,
    RKSResponseDiagnostics,
    RKSResponseError,
    validate_rks_reference,
)
from .rks_zvector import RKSDeePHFZVectorGradients
from .uks_gradient import UKSDeePHFGradients
from .uks_method import UKSDeePHF
from .uks_zvector import UKSDeePHFZVectorGradients
from .pyscf_uks import (
    UKSAdjoint,
    UKSAdjointAdapter,
    UKSAdjointDiagnostics,
    UKSAdjointError,
    UKSNativeGradient,
    UKSResponse,
    UKSResponseAdapter,
    UKSResponseDiagnostics,
    UKSResponseError,
    validate_uks_reference,
)
from .zvector import RHFDeePHFZVectorGradients
from .workflow import build_reference, evaluate_molecule, make_deephf

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
    "RHFBlockedResponseSummary",
    "RHFResponseDiagnostics",
    "RHFResponseError",
    "RHFRootSnapshot",
    "RHFScannerReferenceError",
    "RHFScannerReferenceFactory",
    "RKSDeePHF",
    "RKSDeePHFGradients",
    "RKSDeePHFZVectorGradients",
    "RKSAdjoint",
    "RKSAdjointAdapter",
    "RKSAdjointDiagnostics",
    "RKSAdjointError",
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
    "UHFDeePHFZVectorGradients",
    "UHFAdjoint",
    "UHFAdjointAdapter",
    "UHFAdjointDiagnostics",
    "UHFAdjointError",
    "UHFResponse",
    "UHFResponseAdapter",
    "UHFResponseDiagnostics",
    "UHFResponseError",
    "UKSDeePHF",
    "UKSDeePHFGradients",
    "UKSDeePHFZVectorGradients",
    "UKSAdjoint",
    "UKSAdjointAdapter",
    "UKSAdjointDiagnostics",
    "UKSAdjointError",
    "UKSNativeGradient",
    "UKSResponse",
    "UKSResponseAdapter",
    "UKSResponseDiagnostics",
    "UKSResponseError",
    "generate_rhf_force_frame",
    "solve_scalar_adjoint",
    "scalar_operator_fingerprint",
    "validate_model",
    "validate_force_model",
    "validate_model_output",
    "validate_reference",
    "validate_rks_reference",
    "validate_uhf_reference",
    "validate_uks_reference",
    "write_rhf_force_dataset",
    "build_reference",
    "evaluate_molecule",
    "make_deephf",
]
