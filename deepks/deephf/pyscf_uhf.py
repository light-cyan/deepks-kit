"""Compatibility facade for pyscf_uhf.py."""

from pyscf.scf import ucphf

from .pyscf_uhf_reference import *
from .pyscf_uhf_reference import (
    _native_unrestricted_gradient,
    _version_series,
    _direct_effective_potential,
    _update_fingerprint_value,
    _immutable_array,
    _validated_float64_array,
    _validated_response_array,
    _cycle_limit,
    _response_real_control,
)
from .pyscf_uhf_response_core import *
from .pyscf_uhf_response_core import _UHFLinearResponseCore
from .pyscf_uhf_response import *
from .pyscf_uhf_adjoint import *
from .pyscf_uhf_adjoint import (
    _UHFScalarAdjointProblem,
)
