"""Reference-neutral scalar adjoint equations and audited linear solves."""

from dataclasses import dataclass, replace
import hashlib
import operator
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .contracts import (
    array_fingerprint as _array_fingerprint,
    dataclass_fingerprint,
    immutable_array as _immutable_array,
)


class AdjointError(RuntimeError):
    """Raised when a scalar adjoint equation fails its strict contract."""


@runtime_checkable
class ScalarAdjointProblem(Protocol):
    """Provide one linear response operator and its transpose action."""

    @property
    def dimension(self) -> int:
        """Return the flattened response-space dimension."""

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply the physical forward response operator ``A``."""

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        """Apply the physical transpose response operator ``A.T``."""


@dataclass(frozen=True)
class AdjointDiagnostics:
    """Independent diagnostics for one transpose-adjoint solve."""

    solver: str
    dimension: int
    solve_count: int
    residual_tolerance: float
    objective_gradient_norm: float
    solution_norm: float
    maximum_residual: float
    residual_rms: float
    iteration_count: int = 1


@dataclass(frozen=True)
class AdjointResult:
    """Immutable solution of ``A.T z = b`` for one scalar objective."""

    operator_fingerprint: str
    integrity_fingerprint: str
    objective_gradient: np.ndarray
    solution: np.ndarray
    residual: np.ndarray
    diagnostics: AdjointDiagnostics


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


def _validated_positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"adjoint {name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"adjoint {name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"adjoint {name} must be positive")
    return result


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


def adjoint_integrity_fingerprint(result: AdjointResult) -> str:
    """Return a digest covering every adjoint field except its own digest."""
    return dataclass_fingerprint(
        result,
        excluded=frozenset({"integrity_fingerprint"}),
    )


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


def _problem_action(action, vector: np.ndarray, dimension: int, name: str) -> np.ndarray:
    """Apply one trusted iterative action without hashing or copying full vectors."""
    action_input = _validated_vector(vector, dimension, f"{name} input").view()
    action_input.setflags(write=False)
    try:
        result = action(action_input)
    except Exception as error:
        if isinstance(error, AdjointError):
            raise
        raise AdjointError(f"{name} failed: {error}") from error
    return _validated_vector(result, dimension, name)


def _matrix_free_operator_fingerprint(
    problem: ScalarAdjointProblem,
    dimension: int,
) -> str:
    """Fingerprint supplied scientific-state evidence without operator probes."""
    supplied_fingerprint = getattr(problem, "operator_fingerprint", None)
    if callable(supplied_fingerprint):
        supplied_fingerprint = supplied_fingerprint()
    if supplied_fingerprint is None:
        dense_operator = getattr(problem, "dense_operator", None)
        if callable(dense_operator):
            supplied_fingerprint = _array_fingerprint(
                _validated_matrix(dense_operator(), dimension)
            )
    if type(supplied_fingerprint) is not str or len(supplied_fingerprint) != 64:
        raise AdjointError(
            "matrix-free problems must supply a SHA-256 operator fingerprint"
        )
    try:
        encoded_fingerprint = bytes.fromhex(supplied_fingerprint)
    except ValueError as error:
        raise AdjointError(
            "matrix-free operator fingerprint must be a SHA-256 hex digest"
        ) from error
    digest = hashlib.sha256()
    digest.update(b"matrix-free-scalar-adjoint-v3")
    digest.update(str(dimension).encode("ascii"))
    digest.update(encoded_fingerprint)
    return digest.hexdigest()


def scalar_operator_fingerprint(
    problem: ScalarAdjointProblem,
    *,
    solver: str = "gmres",
) -> str:
    """Return the operator fingerprint used by one adjoint solver backend."""
    dimension = _validated_dimension(problem.dimension)
    if solver == "dense":
        dense_operator = getattr(problem, "dense_operator", None)
        if not callable(dense_operator):
            raise AdjointError(
                "the dense adjoint solver requires a dense_operator action"
            )
        matrix = _validated_matrix(dense_operator(), dimension)
        return _array_fingerprint(matrix)
    if solver == "gmres":
        return _matrix_free_operator_fingerprint(problem, dimension)
    raise ValueError("adjoint solver must be 'dense' or 'gmres'")


@dataclass(frozen=True)
class _LinearSolve:
    operator_fingerprint: str
    solution: np.ndarray
    solver_name: str
    iteration_count: int
    matrix: np.ndarray | None = None


def _solve_dense_adjoint(problem, objective_gradient, dimension) -> _LinearSolve:
    dense_operator = getattr(problem, "dense_operator", None)
    if not callable(dense_operator):
        raise AdjointError(
            "the dense adjoint solver requires a dense_operator action"
        )
    matrix = _immutable_array(_validated_matrix(dense_operator(), dimension))
    try:
        solution = np.linalg.solve(matrix.T, objective_gradient)
    except np.linalg.LinAlgError as error:
        raise AdjointError(f"dense transpose adjoint solve failed: {error}") from error
    except Exception as error:
        raise AdjointError(
            f"dense transpose adjoint solver raised an error: {error}"
        ) from error
    return _LinearSolve(
        operator_fingerprint=_array_fingerprint(matrix),
        solution=solution,
        solver_name="numpy.linalg.solve(A.T, b)",
        iteration_count=1,
        matrix=matrix,
    )


def _linear_operator(action, dimension, name):
    def checked_action(vector):
        return _problem_action(
            action,
            np.asarray(vector, dtype=np.float64),
            dimension,
            name,
        )

    return LinearOperator(
        (dimension, dimension),
        matvec=checked_action,
        dtype=np.float64,
    )


def _solve_matrix_free_adjoint(
    problem,
    objective_gradient,
    dimension,
    residual_tolerance,
    max_cycle,
    restart,
) -> _LinearSolve:
    operator_fingerprint = _matrix_free_operator_fingerprint(problem, dimension)
    linear_operator = _linear_operator(
        problem.apply_transpose,
        dimension,
        "matrix-free transpose adjoint action",
    )
    precondition = getattr(problem, "precondition", None)
    preconditioner = (
        _linear_operator(precondition, dimension, "adjoint preconditioner action")
        if callable(precondition)
        else None
    )
    iteration_count = 0

    def count_iteration(_residual):
        nonlocal iteration_count
        iteration_count += 1

    try:
        solution, convergence_info = gmres(
            linear_operator,
            objective_gradient,
            rtol=0.0,
            atol=min(residual_tolerance, 1.0e-12),
            restart=min(restart, dimension),
            maxiter=max_cycle,
            M=preconditioner,
            callback=count_iteration,
            callback_type="pr_norm",
        )
    except Exception as error:
        raise AdjointError(
            f"matrix-free GMRES adjoint solver raised an error: {error}"
        ) from error
    if convergence_info > 0:
        raise AdjointError(
            "matrix-free GMRES adjoint solve did not converge within "
            f"{max_cycle} restart cycles"
        )
    if convergence_info < 0:
        raise AdjointError("matrix-free GMRES adjoint solve broke down")
    return _LinearSolve(
        operator_fingerprint=operator_fingerprint,
        solution=solution,
        solver_name="scipy.sparse.linalg.gmres(A.T, b)",
        iteration_count=iteration_count,
    )


def _independent_residual(problem, solved, objective_gradient, dimension, solver):
    if _validated_dimension(problem.dimension) != dimension:
        raise AdjointError(
            "adjoint problem dimension changed during independent residual checks"
        )
    self_adjoint = getattr(problem, "is_self_adjoint", None) is True
    if solver == "dense":
        final_matrix = _validated_matrix(problem.dense_operator(), dimension)
        if not np.array_equal(final_matrix, solved.matrix):
            raise AdjointError(
                "adjoint response operator changed during independent residual checks"
            )
        residual_matrix = final_matrix if self_adjoint else final_matrix.T
        return residual_matrix @ solved.solution - objective_gradient, self_adjoint
    residual_action = problem.apply if self_adjoint else problem.apply_transpose
    residual_image = _isolated_problem_action(
        residual_action,
        solved.solution,
        dimension,
        action_name="independent adjoint residual action",
        result_name="independent adjoint residual action",
    )
    if _matrix_free_operator_fingerprint(problem, dimension) != solved.operator_fingerprint:
        raise AdjointError(
            "adjoint response operator changed during independent residual checks"
        )
    return residual_image - objective_gradient, self_adjoint


def _publish_adjoint_result(
    solved,
    objective_gradient,
    residual,
    dimension,
    residual_tolerance,
):
    maximum_residual, residual_rms = _residual_statistics(residual)
    if maximum_residual > residual_tolerance:
        raise AdjointError(
            "adjoint solver residual exceeds tolerance: "
            f"{maximum_residual:.3e} > {residual_tolerance:.3e}"
        )
    objective_gradient_norm = float(np.linalg.norm(objective_gradient))
    solution_norm = float(np.linalg.norm(solved.solution))
    if not np.isfinite(
        (objective_gradient_norm, solution_norm, maximum_residual, residual_rms)
    ).all():
        raise AdjointError("adjoint diagnostics must be finite")
    diagnostics = AdjointDiagnostics(
        solver=solved.solver_name,
        dimension=dimension,
        solve_count=1,
        residual_tolerance=residual_tolerance,
        objective_gradient_norm=objective_gradient_norm,
        solution_norm=solution_norm,
        maximum_residual=maximum_residual,
        residual_rms=residual_rms,
        iteration_count=solved.iteration_count,
    )
    result = AdjointResult(
        operator_fingerprint=solved.operator_fingerprint,
        integrity_fingerprint="",
        objective_gradient=objective_gradient,
        solution=solved.solution,
        residual=residual,
        diagnostics=diagnostics,
    )
    return replace(result, integrity_fingerprint=adjoint_integrity_fingerprint(result))


def solve_scalar_adjoint(
    problem: ScalarAdjointProblem,
    objective_gradient: np.ndarray,
    *,
    residual_tolerance: float = 1.0e-9,
    require_physical_residual: bool = False,
    solver: str = "gmres",
    max_cycle: int = 100,
    restart: int = 50,
) -> AdjointResult:
    """Solve ``A.T z = b`` and retain one independently evaluated residual."""
    if not isinstance(problem, ScalarAdjointProblem):
        raise AdjointError(
            "adjoint problem does not implement the scalar adjoint protocol"
        )
    dimension = _validated_dimension(problem.dimension)
    residual_tolerance = _validated_residual_tolerance(residual_tolerance)
    if not isinstance(require_physical_residual, (bool, np.bool_)):
        raise TypeError("require_physical_residual must be boolean")
    if type(solver) is not str or solver not in {"dense", "gmres"}:
        raise ValueError("adjoint solver must be 'dense' or 'gmres'")
    max_cycle = _validated_positive_integer(max_cycle, "max_cycle")
    restart = _validated_positive_integer(restart, "restart")
    objective_gradient = _immutable_array(
        _validated_vector(
            objective_gradient,
            dimension,
            "adjoint objective gradient",
        )
    )
    solved = (
        _solve_dense_adjoint(problem, objective_gradient, dimension)
        if solver == "dense"
        else _solve_matrix_free_adjoint(
            problem,
            objective_gradient,
            dimension,
            residual_tolerance,
            max_cycle,
            restart,
        )
    )
    solved = replace(
        solved,
        solution=_immutable_array(
            _validated_vector(solved.solution, dimension, "adjoint solution")
        ),
    )
    residual_value, self_adjoint = _independent_residual(
        problem,
        solved,
        objective_gradient,
        dimension,
        solver,
    )
    residual = _immutable_array(
        _validated_vector(
            residual_value,
            dimension,
            "independent adjoint residual",
        )
    )
    if require_physical_residual and not self_adjoint:
        physical = _isolated_problem_action(
            problem.apply,
            solved.solution,
            dimension,
            action_name="physical adjoint audit action",
            result_name="physical adjoint audit action",
        ) - objective_gradient
        if _residual_statistics(physical)[0] > residual_tolerance:
            raise AdjointError("physical adjoint residual exceeds tolerance")
    return _publish_adjoint_result(
        solved,
        objective_gradient,
        residual,
        dimension,
        residual_tolerance,
    )
