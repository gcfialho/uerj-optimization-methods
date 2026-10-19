from __future__ import annotations

import math

import numpy as np
import pytest

from tank_optimizer.objective import (
    SPEC,
    TankSpec,
    barrier_problem,
    constraints,
    cost,
    cost_gradient,
    is_feasible,
    mass,
    penalty_problem,
    volume,
    weld_length,
)


def test_mass_and_cost_known_value() -> None:
    """D = L = 1 m, worked by hand with t = 0.03 m, rho = 8000 kg/m^3."""
    shell = math.pi * 1.0 * (0.53**2 - 0.5**2)  # 0.097075 m^3
    plates = 2 * math.pi * 0.53**2 * 0.03  # 0.052949 m^3
    expected_mass = 8000 * (shell + plates)  # 1200.189 kg
    expected_cost = 4.5 * expected_mass + 20 * 4 * math.pi * 1.03  # 5659.717 $

    assert float(mass(1.0, 1.0)) == pytest.approx(expected_mass)
    assert float(mass(1.0, 1.0)) == pytest.approx(1200.189, abs=1e-3)
    assert float(weld_length(1.0)) == pytest.approx(12.943, abs=1e-3)
    assert float(cost(1.0, 1.0)) == pytest.approx(expected_cost)
    assert float(cost(1.0, 1.0)) == pytest.approx(5659.717, abs=1e-3)


def test_cost_is_vectorized() -> None:
    D, L = np.meshgrid(np.linspace(0.2, 1.0, 5), np.linspace(0.3, 2.0, 4))
    grid = cost(D, L)
    assert grid.shape == D.shape
    assert grid[2, 3] == pytest.approx(float(cost(D[2, 3], L[2, 3])))


def test_cost_increases_with_size() -> None:
    assert cost(0.8, 1.5) > cost(0.7, 1.5)
    assert cost(0.8, 1.5) > cost(0.8, 1.4)


def test_constraints_signs() -> None:
    D = 1.0
    L_nominal = SPEC.volume / (math.pi * D**2 / 4)  # exactly V0
    assert float(volume(D, L_nominal)) == pytest.approx(SPEC.volume)
    assert np.all(constraints(D, L_nominal) <= 0)
    assert is_feasible(D, L_nominal)

    g_small = constraints(0.5, 0.5)  # V = 0.098 m^3 < 0.72
    assert g_small[0] > 0
    assert g_small[1] < 0
    assert not is_feasible(0.5, 0.5)

    g_large = constraints(1.1, 1.5)  # V = 1.43 m^3 > 0.88 and D > D_max
    assert g_large[1] > 0
    assert g_large[3] > 0
    assert constraints(0.7, 2.5)[2] > 0  # L > L_max


def test_barrier_problem_values() -> None:
    phi = barrier_problem(1e-3)
    assert phi.fun(np.array([0.0, 1.0])) == math.inf
    assert phi.fun(np.array([0.5, 0.5])) == math.inf  # infeasible (V too small)

    x = np.array([0.98, 1.0])  # V = 0.754 m^3, strictly feasible
    expected = float(cost(*x)) - 1e-3 * float(np.sum(np.log(-constraints(*x))))
    assert phi.fun(x) == pytest.approx(expected)
    assert abs(phi.fun(x) - float(cost(*x))) < 0.05


def test_penalty_problem_values() -> None:
    psi = penalty_problem(1e6, margin=0.0)
    assert psi.fun(np.array([-0.1, 1.0])) == math.inf
    x = np.array([1.0, 1.0])
    assert psi.fun(x) == pytest.approx(float(cost(*x)))  # no violation, no penalty
    g1 = SPEC.min_volume - float(volume(0.5, 0.5))
    assert psi.fun(np.array([0.5, 0.5])) == pytest.approx(float(cost(0.5, 0.5)) + 1e6 * g1**2)


def _fd_grad(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    return np.array([(f(x + e) - f(x - e)) / (2 * h) for e in h * np.eye(x.size)])


@pytest.mark.parametrize(
    ("problem", "x"),
    [
        (barrier_problem(1.0), [0.9, 1.2]),
        (barrier_problem(0.1), [0.97, 1.0]),
        (penalty_problem(1e5, 1e-3), [0.5, 0.5]),  # volume constraint violated
        (penalty_problem(1e5, 1e-3), [1.1, 1.5]),  # V_max and D_max violated
        (penalty_problem(1e5, 1e-3), [0.9, 1.2]),  # feasible: plain cost
    ],
)
def test_exact_derivatives_match_finite_differences(problem, x: list[float]) -> None:
    x = np.array(x)
    np.testing.assert_allclose(problem.grad(x), _fd_grad(problem.fun, x), rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(problem.hess(x), _fd_grad(problem.grad, x).T, rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(problem.hess(x), problem.hess(x).T)


def test_cost_gradient_is_positive() -> None:
    """A larger tank always costs more, so the volume lower bound is active at the optimum."""
    D, L = np.meshgrid(np.linspace(0.1, 1.0, 10), np.linspace(0.1, 2.0, 10))
    for d, l_ in zip(D.ravel(), L.ravel(), strict=True):
        assert np.all(cost_gradient(d, l_) > 0)


def test_custom_spec() -> None:
    spec = TankSpec(volume=1.0, thickness=0.01)
    assert spec.min_volume == pytest.approx(0.9)
    assert spec.max_volume == pytest.approx(1.1)
    assert cost(1.0, 1.0, spec) < cost(1.0, 1.0)
