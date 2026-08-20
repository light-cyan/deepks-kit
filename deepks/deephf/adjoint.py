"""Reference-neutral scalar adjoint equations and strict dense solves."""

from dataclasses import dataclass, fields, replace
import hashlib
import operator
from typing import Protocol, runtime_checkable

import numpy as np


class AdjointError(RuntimeError):
    """Raised when a scalar adjoint equation fails its strict contract."""


@runtime_checkable
class ScalarAdjointProblem(Protocol):
    """Provide one linear response operator and its transpose action."""

    @property
    def dimension(self) -> int:
        """Return the flattened response-space dimension."""

    def dense_operator(self) -> np.ndarray:
        """Return the response matrix ``A`` in ``A x = -B``."""

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply the physical forward response operator ``A``."""

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        """Apply the physical transpose response operator ``A.T``."""


@dataclass(frozen=True)
class AdjointDiagnostics:
    """Independent diagnostics for one literal dense transpose solve."""

    solver: str
    dimension: int
    solve_count: int
    residual_tolerance: float
    objective_gradient_norm: float
    solution_norm: float
    maximum_solver_residual: float
    solver_residual_rms: float
    maximum_transpose_residual: float
    transpose_residual_rms: float
    maximum_physical_residual: float
    physical_residual_rms: float


@dataclass(frozen=True)
class AdjointResult:
    """Immutable solution of ``A.T z = b`` for one scalar objective."""

    operator_fingerprint: str
    integrity_fingerprint: str
    objective_gradient: np.ndarray
    solution: np.ndarray
    solver_residual: np.ndarray
    transpose_residual: np.ndarray
    physical_residual: np.ndarray
    diagnostics: AdjointDiagnostics


def _immutable_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _validated_dimension(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise AdjointError("adjoint response dimension must be an integer")
    try:
        dimension = operator.index(value)
    except TypeError as error:
        raise AdjointError("adjoint response dimension must be an integer") from error
    if dimension <= 0:
        raise AdjointError("adjoint response dimension must be positive")
    return dimension


def _validated_residual_tolerance(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("adjoint residual_tolerance must be a real number")
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "adjoint residual_tolerance must be finite and positive"
        )
    return tolerance


def _validated_vector(value, dimension: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise AdjointError(f"{name} is not a numerical array: {error}") from error
    if array.shape != (dimension,):
        raise AdjointError(
            f"{name} has shape {array.shape}; expected {(dimension,)}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise AdjointError(f"{name} must be a real float64 array")
    if not np.isfinite(array).all():
        raise AdjointError(f"{name} must be finite")
    return array


def _validated_matrix(value, dimension: int) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except Exception as error:
        raise AdjointError(
            f"adjoint response operator is not a numerical array: {error}"
        ) from error
    expected_shape = (dimension, dimension)
    if matrix.shape != expected_shape:
        raise AdjointError(
            "adjoint response operator has shape "
            f"{matrix.shape}; expected {expected_shape}"
        )
    if matrix.dtype != np.dtype(np.float64) or np.iscomplexobj(matrix):
        raise AdjointError(
            "adjoint response operator must be a real float64 array"
        )
    if not np.isfinite(matrix).all():
        raise AdjointError("adjoint response operator must be finite")
    return matrix


def _array_fingerprint(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def adjoint_integrity_fingerprint(result: AdjointResult) -> str:
    """Return a digest covering every adjoint field except its own digest."""
    digest = hashlib.sha256()
    for field in fields(result):
        if field.name == "integrity_fingerprint":
            continue
        value = getattr(result, field.name)
        digest.update(field.name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _residual_statistics(residual: np.ndarray) -> tuple[float, float]:
    return (
        float(np.max(np.abs(residual), initial=0.0)),
        float(np.sqrt(np.mean(np.square(residual)))),
    )


def _action_attempted_input_mutation(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "read-only",
            "readonly",
            "writeable flag",
            "writable flag",
        )
    )


def _isolated_problem_action(
    action,
    solution: np.ndarray,
    dimension: int,
    *,
    action_name: str,
    result_name: str,
) -> np.ndarray:
    """Apply an untrusted action to a private immutable solution snapshot."""
    action_input = _immutable_array(
        _validated_vector(
            solution,
            dimension,
            f"{action_name} input",
        )
    )
    input_fingerprint = _array_fingerprint(action_input)
    try:
        action_result = action(action_input)
    except Exception as error:
        input_changed = (
            action_input.flags.writeable
            or _array_fingerprint(action_input) != input_fingerprint
        )
        if input_changed:
            raise AdjointError(
                f"{action_name} mutated its isolated adjoint solution input"
            ) from error
        if _action_attempted_input_mutation(error):
            raise AdjointError(
                f"{action_name} attempted to mutate its immutable adjoint "
                "solution input"
            ) from error
        if isinstance(error, AdjointError):
            raise
        raise AdjointError(f"{action_name} failed: {error}") from error
    input_changed = (
        action_input.flags.writeable
        or _array_fingerprint(action_input) != input_fingerprint
    )
    if input_changed:
        raise AdjointError(
            f"{action_name} mutated its isolated adjoint solution input"
        )
    action_result = _immutable_array(
        _validated_vector(action_result, dimension, result_name)
    )
    if (
        action_input.flags.writeable
        or _array_fingerprint(action_input) != input_fingerprint
    ):
        raise AdjointError(
            f"{action_name} mutated its isolated adjoint solution input"
        )
    return action_result


def solve_scalar_adjoint(
    problem: ScalarAdjointProblem,
    objective_gradient: np.ndarray,
    *,
    residual_tolerance: float = 1.0e-9,
    require_physical_residual: bool = False,
) -> AdjointResult:
    """Solve ``A.T z = b`` once and audit literal and independent actions."""
    if not isinstance(problem, ScalarAdjointProblem):
        raise AdjointError(
            "adjoint problem does not implement the scalar adjoint protocol"
        )
    dimension = _validated_dimension(problem.dimension)
    residual_tolerance = _validated_residual_tolerance(residual_tolerance)
    if not isinstance(require_physical_residual, (bool, np.bool_)):
        raise TypeError("require_physical_residual must be boolean")
    objective_gradient = _immutable_array(
        _validated_vector(
            objective_gradient,
            dimension,
            "adjoint objective gradient",
        )
    )
    matrix = _immutable_array(
        _validated_matrix(problem.dense_operator(), dimension)
    )
    operator_fingerprint = _array_fingerprint(matrix)
    try:
        solution = np.linalg.solve(matrix.T, objective_gradient)
    except np.linalg.LinAlgError as error:
        raise AdjointError(
            f"dense transpose adjoint solve failed: {error}"
        ) from error
    except Exception as error:
        raise AdjointError(
            f"dense transpose adjoint solver raised an error: {error}"
        ) from error
    solution = _immutable_array(
        _validated_vector(solution, dimension, "adjoint solution")
    )
    transpose_image = _isolated_problem_action(
        problem.apply_transpose,
        solution,
        dimension,
        action_name="independent transpose adjoint action",
        result_name="independent transpose adjoint action",
    )
    physical_image = _isolated_problem_action(
        problem.apply,
        solution,
        dimension,
        action_name="physical adjoint action",
        result_name="physical adjoint action",
    )
    final_dimension = _validated_dimension(problem.dimension)
    if final_dimension != dimension:
        raise AdjointError(
            "adjoint problem dimension changed during independent residual checks"
        )
    final_matrix = _validated_matrix(problem.dense_operator(), dimension)
    if not np.array_equal(final_matrix, matrix):
        raise AdjointError(
            "adjoint response operator changed during independent residual checks"
        )
    solver_residual = _immutable_array(
        _validated_vector(
            matrix.T @ solution - objective_gradient,
            dimension,
            "literal transpose adjoint residual",
        )
    )
    transpose_residual = _immutable_array(
        _validated_vector(
            transpose_image - objective_gradient,
            dimension,
            "independent transpose adjoint residual",
        )
    )
    physical_residual = _immutable_array(
        _validated_vector(
            physical_image - objective_gradient,
            dimension,
            "physical adjoint residual",
        )
    )
    maximum_solver_residual, solver_residual_rms = _residual_statistics(
        solver_residual
    )
    maximum_transpose_residual, transpose_residual_rms = _residual_statistics(
        transpose_residual
    )
    maximum_physical_residual, physical_residual_rms = _residual_statistics(
        physical_residual
    )
    if maximum_solver_residual > residual_tolerance:
        raise AdjointError(
            "literal transpose adjoint residual exceeds tolerance: "
            f"{maximum_solver_residual:.3e} > {residual_tolerance:.3e}"
        )
    if maximum_transpose_residual > residual_tolerance:
        raise AdjointError(
            "independent transpose adjoint residual exceeds tolerance: "
            f"{maximum_transpose_residual:.3e} > {residual_tolerance:.3e}"
        )
    if require_physical_residual and maximum_physical_residual > residual_tolerance:
        raise AdjointError(
            "physical adjoint residual exceeds tolerance: "
            f"{maximum_physical_residual:.3e} > {residual_tolerance:.3e}"
        )
    objective_gradient_norm = float(np.linalg.norm(objective_gradient))
    solution_norm = float(np.linalg.norm(solution))
    diagnostic_values = (
        objective_gradient_norm,
        solution_norm,
        maximum_solver_residual,
        solver_residual_rms,
        maximum_transpose_residual,
        transpose_residual_rms,
        maximum_physical_residual,
        physical_residual_rms,
    )
    if not np.isfinite(diagnostic_values).all():
        raise AdjointError("adjoint diagnostics must be finite")
    diagnostics = AdjointDiagnostics(
        solver="numpy.linalg.solve(A.T, b)",
        dimension=dimension,
        solve_count=1,
        residual_tolerance=residual_tolerance,
        objective_gradient_norm=objective_gradient_norm,
        solution_norm=solution_norm,
        maximum_solver_residual=maximum_solver_residual,
        solver_residual_rms=solver_residual_rms,
        maximum_transpose_residual=maximum_transpose_residual,
        transpose_residual_rms=transpose_residual_rms,
        maximum_physical_residual=maximum_physical_residual,
        physical_residual_rms=physical_residual_rms,
    )
    result = AdjointResult(
        operator_fingerprint=operator_fingerprint,
        integrity_fingerprint="",
        objective_gradient=_immutable_array(objective_gradient),
        solution=_immutable_array(solution),
        solver_residual=_immutable_array(solver_residual),
        transpose_residual=_immutable_array(transpose_residual),
        physical_residual=_immutable_array(physical_residual),
        diagnostics=diagnostics,
    )
    return replace(
        result,
        integrity_fingerprint=adjoint_integrity_fingerprint(result),
    )
