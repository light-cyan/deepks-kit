"""Perturbative DeePHF energy and strict RHF analytic-gradient methods."""

from .adjoint import (
    AdjointDiagnostics,
    AdjointError,
    AdjointResult,
    ScalarAdjointProblem,
    solve_scalar_adjoint,
)

from .capabilities import (
    DeePHFCapabilityError,
    validate_model,
    validate_model_output,
    validate_reference,
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
)
from .scanner import RHFDeePHFGradientScanner, RHFDeePHFScannerError
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
    "ScalarAdjointProblem",
    "generate_rhf_force_frame",
    "solve_scalar_adjoint",
    "validate_model",
    "validate_model_output",
    "validate_reference",
    "write_rhf_force_dataset",
]
