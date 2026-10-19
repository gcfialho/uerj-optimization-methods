from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from tank_optimizer.constrained import solve
from tank_optimizer.objective import SPEC, cost, is_feasible, volume
from tank_optimizer.optimizers import DFP, METHODS, Objective, minimize

A = np.array([[3.0, 1.0], [1.0, 2.0]])
B = np.array([1.0, -1.0])
X_STAR = np.linalg.solve(A, B)


def quadratic(x: np.ndarray) -> float:
    return float(0.5 * x @ A @ x - B @ x)


def rosenbrock(x: np.ndarray) -> float:
    return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def test_finite_differences_are_accurate() -> None:
    f = Objective(quadratic)
    x = np.array([0.3, -0.7])
    np.testing.assert_allclose(f.grad(x), A @ x - B, atol=1e-6)
    np.testing.assert_allclose(f.hessian(x), A, atol=1e-4)
    assert f.nfev == 2 * 2 + 4 * 3  # 2 per gradient entry, 4 per upper-triangle Hessian entry
    assert (f.ngev, f.nhev) == (1, 1)


def test_hessian_from_exact_gradient() -> None:
    f = Objective(quadratic, grad=lambda x: A @ x - B)
    np.testing.assert_allclose(f.hessian(np.array([0.3, -0.7])), A, atol=1e-6)
    assert f.nfev == 0
    assert f.ngev == 4


def test_exact_derivatives_are_used_when_given() -> None:
    result = minimize(
        quadratic, [2.0, 2.0], method="Newton", grad=lambda x: A @ x - B, hess=lambda x: A
    )
    np.testing.assert_allclose(result.x, X_STAR, atol=1e-12)
    assert result.nit == 1
    assert result.nhev == 1


@pytest.mark.parametrize("method", sorted(METHODS))
def test_methods_solve_a_convex_quadratic(method: str) -> None:
    result = minimize(quadratic, [2.0, 2.0], method=method, tol=1e-6, max_iter=500)
    assert result.converged, result.message
    np.testing.assert_allclose(result.x, X_STAR, atol=1e-5)
    assert result.grad_norms[-1] <= 1e-6
    assert len(result.history) == len(result.grad_norms) == result.nit + 1


@pytest.mark.parametrize("method", ["Newton", "DFP"])
def test_second_order_methods_solve_rosenbrock(method: str) -> None:
    result = minimize(rosenbrock, [-1.2, 1.0], method=method, tol=1e-5, max_iter=500)
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)


def test_dfp_keeps_inverse_hessian_positive_definite() -> None:
    opt = DFP(Objective(rosenbrock), [-1.2, 1.0], tol=1e-5, max_iter=50)
    updates = []
    original = opt.update

    def spy(s: np.ndarray, y: np.ndarray) -> None:
        original(s, y)
        updates.append(opt.B.copy())

    opt.update = spy  # type: ignore[method-assign]
    opt.optimize()
    assert updates
    for B in updates:
        np.testing.assert_allclose(B, B.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(B) > 0)


def test_every_step_satisfies_armijo() -> None:
    result = minimize(rosenbrock, [-1.2, 1.0], method="SD", max_iter=50)
    values = [rosenbrock(x) for x in result.history]
    assert all(b <= a for a, b in itertools.pairwise(values))


def test_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        minimize(quadratic, [0.0, 0.0], method="BFGS")


def test_stop_when_callback() -> None:
    result = minimize(rosenbrock, [-1.2, 1.0], method="SD", max_iter=500)
    opt = METHODS["SD"](Objective(rosenbrock), [-1.2, 1.0], stop_when=lambda x: x[0] > -1.0)
    stopped = opt.optimize()
    assert stopped.converged
    assert stopped.message == "Stopping condition met."
    assert stopped.nit < result.nit


# --- the tank problem -----------------------------------------------------------


@pytest.fixture(scope="module")
def reference_optimum() -> tuple[np.ndarray, float]:
    """Cost increases with D and L, so the optimum lies on V = V_min: a 1-D problem in D."""

    def length(D: float) -> float:
        return 4 * SPEC.min_volume / (np.pi * D**2)

    res = minimize_scalar(
        lambda D: float(cost(D, length(D))),
        bounds=(np.sqrt(4 * SPEC.min_volume / (np.pi * SPEC.max_length)), SPEC.max_diameter),
        method="bounded",
        options={"xatol": 1e-12},
    )
    x = np.array([res.x, length(res.x)])
    assert is_feasible(*x, tol=1e-12)
    return x, float(res.fun)


def test_reference_optimum_known_value(reference_optimum: tuple[np.ndarray, float]) -> None:
    x, c = reference_optimum
    np.testing.assert_allclose(x, [0.948760, 1.018428], atol=1e-6)
    assert c == pytest.approx(5354.349, abs=1e-3)
    assert float(volume(*x)) == pytest.approx(SPEC.min_volume)


STARTS = [(0.5, 0.5), (0.45, 0.55), (0.98, 0.95), (0.8, 1.2), (1.1, 1.5), (0.32, 1.98)]


@pytest.mark.parametrize("start", STARTS)
def test_newton_converges_to_the_constrained_optimum(
    start: tuple[float, float], reference_optimum: tuple[np.ndarray, float]
) -> None:
    x_ref, cost_ref = reference_optimum
    result = solve(start, method="Newton", tol=1e-6, max_iter=200)
    assert result.converged, result.message
    assert result.feasible
    # The final barrier weight (1e-2) keeps the design marginally inside: O(mu) in cost.
    assert result.cost == pytest.approx(cost_ref, abs=0.05)
    np.testing.assert_allclose(result.x, x_ref, atol=1e-4)
    assert result.grad_norms[-1] <= 1e-6


@pytest.mark.parametrize("start", STARTS)
def test_dfp_reaches_the_constrained_optimum(
    start: tuple[float, float], reference_optimum: tuple[np.ndarray, float]
) -> None:
    x_ref, cost_ref = reference_optimum
    result = solve(start, method="DFP", tol=1e-6, max_iter=200)
    assert result.feasible
    assert result.cost == pytest.approx(cost_ref, abs=0.05)
    np.testing.assert_allclose(result.x, x_ref, atol=1e-3)


@pytest.mark.parametrize("start", STARTS)
def test_steepest_descent_ends_feasible(
    start: tuple[float, float], reference_optimum: tuple[np.ndarray, float]
) -> None:
    """SD is far too slow for the ill-conditioned barrier problems: feasible, not optimal."""
    _, cost_ref = reference_optimum
    result = solve(start, method="SD", tol=1e-6, max_iter=200)
    assert result.feasible
    assert cost_ref <= result.cost < 1.07 * cost_ref


def test_solve_bookkeeping() -> None:
    result = solve((0.5, 0.5), method="Newton")
    assert result.stages[0].kind == "penalty"  # infeasible start -> phase 1 first
    assert result.stages[-1].kind == "barrier"
    assert sum(st.nit for st in result.stages) == result.nit
    assert len(result.history) == len(result.grad_norms) == result.nit + 1
    assert result.nhev == result.nit  # one exact Hessian per Newton iteration

    feasible_start = solve((0.95, 1.05), method="Newton")
    assert all(st.kind == "barrier" for st in feasible_start.stages)


def test_solve_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        solve((0.5, 0.5), method="BFGS")
