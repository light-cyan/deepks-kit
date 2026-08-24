"""Compatibility facade for pyscf_rhf.py."""

import pyscf
from pyscf.scf import cphf

from .pyscf_rhf_reference import *
from .pyscf_rhf_reference import (
    _version_series,
    _immutable_metadata,
    _immutable_array,
    _update_fingerprint_value,
    _molecule_static_fingerprint,
    _reject_molecule_instance_callables,
    _root_integrity_fingerprint,
    _SCANNER_INITIAL_GUESSES,
    _scanner_real_control,
    _scanner_integer_control,
    _scanner_boolean_control,
    _scanner_scf_controls,
)
from .pyscf_rhf_scanner import *
from .pyscf_rhf_scanner import (
    _cycle_limit,
    _adjoint_real_control,
    _validated_float64_array,
)
from .pyscf_rhf_response import *
from .pyscf_rhf_response import (
    _RHFLinearResponseCore,
)
from .pyscf_rhf_adjoint import *
from .pyscf_rhf_adjoint import (
    _RHFScalarAdjointProblem,
)
