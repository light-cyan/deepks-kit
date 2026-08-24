"""Compatibility facade for pyscf_rks.py."""

from pyscf.grad import rks as rks_grad
from pyscf.scf import cphf

from .pyscf_dft_provenance import *
from .pyscf_dft_provenance import (
    _SUPPORTED_RADI_METHOD,
    _SUPPORTED_RADII_ADJUST,
    _SUPPORTED_BECKE_SCHEME,
    _SUPPORTED_GRIDS_RESPONSE,
    _SUPPORTED_NUMINT_IMPLEMENTATIONS,
    _SUPPORTED_LIBXC_IMPLEMENTATIONS,
    _UKS_REFERENCE_TYPE,
    _SUPPORTED_REFERENCE_IMPLEMENTATIONS,
    _static_callable_definitions,
    _NATIVE_RKS_GRADIENT_METHODS,
    _SUPPORTED_NATIVE_RKS_GRADIENT,
    _GRID_WEIGHT_FD_STEP,
    _GRID_WEIGHT_DERIVATIVE_ATOL,
    _GRID_WEIGHT_DERIVATIVE_RTOL,
    _GRID_RESPONSE_WEIGHT_ATOL,
    _VALIDATED_RKS_REFERENCES,
    _GRID_PROVENANCE_CACHE,
    _version_series,
    _array_fingerprint,
    _qualified_name,
    _validated_float64_array,
    _cycle_limit,
    _response_real_control,
    _normalized_functional_components,
    _validate_dft_implementations,
    _evaluate_libxc_cache,
    _functional_provenance,
    _normalized_atom_grid,
    _grid_arrays,
    _build_strict_grid,
    _finite_difference_grid_weight_derivative,
    _validated_grid_response_blocks,
    _build_grid_provenance,
    _grid_provenance,
)
from .pyscf_rks_reference import *
from .pyscf_rks_reference import (
    _dense_ground_state_lda_quadrature,
    _audit_rks_reference,
    _dft_reference_validation_fingerprint,
)
from .pyscf_rks_response_core import *
from .pyscf_rks_response_core import (
    _RKSLinearResponseProblem,
    _RKSLinearResponseCore,
)
from .pyscf_rks_response import *
from .pyscf_rks_adjoint import *
from .pyscf_rks_native import *
