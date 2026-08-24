"""Reference-neutral scalar adjoint equations and audited linear solves."""

from dataclasses import dataclass, fields, replace
import hashlib
import operator
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


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
    matrix = None
    iteration_count = 1
    if solver == "dense":
        dense_operator = getattr(problem, "dense_operator", None)
        if not callable(dense_operator):
            raise AdjointError(
                "the dense adjoint solver requires a dense_operator action"
            )
        matrix = _immutable_array(
            _validated_matrix(dense_operator(), dimension)
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
        solver_name = "numpy.linalg.solve(A.T, b)"
    else:
        operator_fingerprint = _matrix_free_operator_fingerprint(
            problem,
            dimension,
        )

        def transpose_action(vector):
            return _problem_action(
                problem.apply_transpose,
                np.asarray(vector, dtype=np.float64),
                dimension,
                "matrix-free transpose adjoint action",
            )

        linear_operator = LinearOperator(
            (dimension, dimension),
            matvec=transpose_action,
            dtype=np.float64,
        )
        precondition = getattr(problem, "precondition", None)
        preconditioner = None
        if callable(precondition):
            def precondition_action(vector):
                return _problem_action(
                    precondition,
                    np.asarray(vector, dtype=np.float64),
                    dimension,
                    "adjoint preconditioner action",
                )

            preconditioner = LinearOperator(
                (dimension, dimension),
                matvec=precondition_action,
                dtype=np.float64,
            )
        iteration_count = 0

        def count_iteration(_residual):
            nonlocal iteration_count
            iteration_count += 1

        try:
            krylov_tolerance = min(residual_tolerance, 1.0e-12)
            solution, convergence_info = gmres(
                linear_operator,
                objective_gradient,
                rtol=0.0,
                atol=krylov_tolerance,
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
        if convergence_info != 0:
            if convergence_info > 0:
                raise AdjointError(
                    "matrix-free GMRES adjoint solve did not converge within "
                    f"{max_cycle} restart cycles"
                )
            raise AdjointError("matrix-free GMRES adjoint solve broke down")
        solver_name = "scipy.sparse.linalg.gmres(A.T, b)"
    solution = _immutable_array(
        _validated_vector(solution, dimension, "adjoint solution")
    )
    final_dimension = _validated_dimension(problem.dimension)
    if final_dimension != dimension:
        raise AdjointError(
            "adjoint problem dimension changed during independent residual checks"
        )
    if solver == "dense":
        final_matrix = _validated_matrix(dense_operator(), dimension)
        if not np.array_equal(final_matrix, matrix):
            raise AdjointError(
                "adjoint response operator changed during independent residual checks"
            )
        residual_value = final_matrix.T @ solution - objective_gradient
    else:
        residual_action = (
            problem.apply
            if getattr(problem, "is_self_adjoint", None) is True
            else problem.apply_transpose
        )
        residual_image = _isolated_problem_action(
            residual_action,
            solution,
            dimension,
            action_name="independent adjoint residual action",
            result_name="independent adjoint residual action",
        )
        final_fingerprint = _matrix_free_operator_fingerprint(
            problem,
            dimension,
        )
        if final_fingerprint != operator_fingerprint:
            raise AdjointError(
                "adjoint response operator changed during independent residual checks"
            )
        residual_value = residual_image - objective_gradient
    residual = _immutable_array(
        _validated_vector(
            residual_value,
            dimension,
            "independent adjoint residual",
        )
    )
    maximum_residual, residual_rms = _residual_statistics(residual)
    if maximum_residual > residual_tolerance:
        residual_label = (
            "literal transpose adjoint residual"
            if solver == "dense"
            else "adjoint solver residual"
        )
        raise AdjointError(
            f"{residual_label} exceeds tolerance: "
            f"{maximum_residual:.3e} > {residual_tolerance:.3e}"
        )
    if require_physical_residual and getattr(problem, "is_self_adjoint", None) is not True:
        physical = _isolated_problem_action(
            problem.apply,
            solution,
            dimension,
            action_name="physical adjoint audit action",
            result_name="physical adjoint audit action",
        ) - objective_gradient
        if _residual_statistics(physical)[0] > residual_tolerance:
            raise AdjointError("physical adjoint residual exceeds tolerance")
    objective_gradient_norm = float(np.linalg.norm(objective_gradient))
    solution_norm = float(np.linalg.norm(solution))
    diagnostic_values = (
        objective_gradient_norm,
        solution_norm,
        maximum_residual,
        residual_rms,
    )
    if not np.isfinite(diagnostic_values).all():
        raise AdjointError("adjoint diagnostics must be finite")
    diagnostics = AdjointDiagnostics(
        solver=solver_name,
        dimension=dimension,
        solve_count=1,
        residual_tolerance=residual_tolerance,
        objective_gradient_norm=objective_gradient_norm,
        solution_norm=solution_norm,
        maximum_residual=maximum_residual,
        residual_rms=residual_rms,
        iteration_count=iteration_count,
    )
    result = AdjointResult(
        operator_fingerprint=operator_fingerprint,
        integrity_fingerprint="",
        objective_gradient=objective_gradient,
        solution=solution,
        residual=residual,
        diagnostics=diagnostics,
    )
    return replace(
        result,
        integrity_fingerprint=adjoint_integrity_fingerprint(result),
    )
