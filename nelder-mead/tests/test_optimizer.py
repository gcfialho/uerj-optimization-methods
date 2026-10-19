from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from nelder_mead import (
    FUNCTIONS,
    PROBLEMS,
    Coefficients,
    Step,
    default_simplex,
    get_problem,
    minimize,
    simplex_volume,
)

# (problem, max distance to the nearest known minimizer)
DEMOS = [
    ("rosenbrock-2d", 1e-6),
    ("booth-2d", 1e-6),
    ("himmelblau-2d", 1e-6),
    ("sphere-3d", 1e-6),
    ("shifted-sphere-3d", 1e-6),
]


def run(slug: str, **kwargs):
    problem = PROBLEMS[slug]
    return problem, minimize(
        problem.function.func, initial_simplex=problem.initial_simplex, **kwargs
    )


@pytest.mark.parametrize(("slug", "tol"), DEMOS)
def test_demo_converges_to_known_minimum(slug: str, tol: float) -> None:
    problem, result = run(slug)
    assert result.converged, result.message
    assert result.fun < 1e-10
    assert problem.function.error(result.x) < tol


@pytest.mark.parametrize("slug", [s for s, _ in DEMOS])
def test_matches_scipy(slug: str) -> None:
    """Same formulation, simplex and tolerances as SciPy -> the same sequence of simplices."""
    problem, result = run(slug)
    ref = scipy_minimize(
        problem.function.func,
        problem.initial_simplex[0],
        method="Nelder-Mead",
        options={"initial_simplex": problem.initial_simplex, "xatol": 1e-8, "fatol": 1e-8},
    )
    assert ref.success
    # SciPy starts its iteration counter at 1, so its `nit` is one more than the number of
    # simplex updates it performed.
    assert result.nit == ref.nit - 1
    assert result.nfev == ref.nfev
    np.testing.assert_allclose(result.x, ref.x, rtol=0, atol=1e-12)
    assert result.fun == pytest.approx(ref.fun, abs=1e-15)


@pytest.mark.parametrize("dim", [2, 4, 6])
def test_higher_dimensional_sphere_and_rosenbrock(dim: int) -> None:
    sphere = get_problem("sphere", dim)
    result = minimize(sphere.function.func, initial_simplex=sphere.initial_simplex)
    assert sphere.function.error(result.x) < 1e-6

    rosen = get_problem("rosenbrock", dim)
    result = minimize(
        rosen.function.func,
        initial_simplex=rosen.initial_simplex,
        max_iter=20_000,
        max_fev=40_000,
    )
    ref = scipy_minimize(
        rosen.function.func,
        rosen.initial_simplex[0],
        method="Nelder-Mead",
        options={
            "initial_simplex": rosen.initial_simplex,
            "xatol": 1e-8,
            "fatol": 1e-8,
            "maxiter": 20_000,
            "maxfev": 40_000,
        },
    )
    assert result.fun == pytest.approx(ref.fun, rel=1e-6, abs=1e-12)


def egg_crate(x: np.ndarray) -> float:
    """Multimodal function whose first iteration from the unit simplex is a shrink."""
    return float(x[0] ** 2 + x[1] ** 2 + 3 * np.cos(5 * x[0]) * np.cos(5 * x[1]))


UNIT_SIMPLEX = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
INVARIANT_CASES = [(PROBLEMS[s].function.func, PROBLEMS[s].initial_simplex) for s, _ in DEMOS]
INVARIANT_CASES.append((egg_crate, UNIT_SIMPLEX))


def test_simplex_invariants() -> None:
    """Sorted vertices, monotone best value and the volume change of every operation."""
    coef = Coefficients()
    seen: set[Step] = set()
    for func, simplex in INVARIANT_CASES:
        result = minimize(func, initial_simplex=simplex)
        n = simplex.shape[1]
        expected_ratio = {
            Step.REFLECT: coef.reflection,
            Step.EXPAND: coef.reflection * coef.expansion,
            Step.CONTRACT_OUTSIDE: coef.reflection * coef.contraction,
            Step.CONTRACT_INSIDE: coef.contraction,
            Step.SHRINK: coef.shrink**n,
        }
        assert result.steps[0] is Step.INITIAL
        assert len(result.simplices) == len(result.fvalues) == len(result.steps) == result.nit + 1
        seen.update(result.steps[1:])
        for k, (sim, fsim) in enumerate(zip(result.simplices, result.fvalues, strict=True)):
            assert sim.shape == (n + 1, n)
            assert np.all(np.diff(fsim) >= 0), "vertices must be sorted by f"
            np.testing.assert_array_equal(fsim, [func(v) for v in sim])
            if k == 0:
                continue
            assert fsim[0] <= result.fvalues[k - 1][0], "best value must never increase"
            ratio = simplex_volume(sim) / simplex_volume(result.simplices[k - 1])
            assert ratio == pytest.approx(expected_ratio[result.steps[k]], rel=1e-6)
    assert seen == set(Step) - {Step.INITIAL}, "every branch must be exercised"


def test_shrink_matches_scipy() -> None:
    result = minimize(egg_crate, initial_simplex=UNIT_SIMPLEX)
    assert result.steps[1] is Step.SHRINK
    ref = scipy_minimize(
        egg_crate,
        UNIT_SIMPLEX[0],
        method="Nelder-Mead",
        options={"initial_simplex": UNIT_SIMPLEX, "xatol": 1e-8, "fatol": 1e-8},
    )
    assert (result.nit, result.nfev) == (ref.nit - 1, ref.nfev)
    np.testing.assert_allclose(result.x, ref.x, rtol=0, atol=1e-12)


def test_step_formulas_on_a_single_iteration() -> None:
    """Hand-checked first iteration on f(x) = x1 + 2 x2 (reflection then expansion)."""

    def linear(x: np.ndarray) -> float:
        return float(x[0] + 2 * x[1])

    simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = minimize(linear, initial_simplex=simplex, max_iter=1)
    # worst = (0, 1), centroid = (0.5, 0), reflected = (1, -1) with f = -1 < f_best = 0,
    # expanded = (1.5, -2) with f = -2.5 < f_reflected -> accept the expansion.
    assert result.steps[1] is Step.EXPAND
    np.testing.assert_allclose(result.simplices[1][0], [1.5, -2.0])
    assert result.nfev == 3 + 2


def test_budget_limits_are_respected() -> None:
    problem = PROBLEMS["rosenbrock-2d"]
    result = minimize(problem.function.func, initial_simplex=problem.initial_simplex, max_iter=10)
    assert not result.converged
    assert result.nit == 10
    result = minimize(problem.function.func, initial_simplex=problem.initial_simplex, max_fev=20)
    assert not result.converged
    assert result.nfev >= 20
    assert result.nfev <= 20 + problem.dim + 1  # at most one iteration past the budget


def test_default_simplex_matches_scipy_convention() -> None:
    sim = default_simplex([1.0, 0.0, -2.0])
    np.testing.assert_allclose(
        sim, [[1, 0, -2], [1.05, 0, -2], [1, 0.00025, -2], [1, 0, -2.1]], rtol=0, atol=1e-15
    )


def test_nan_is_treated_as_infinity() -> None:
    def f(x: np.ndarray) -> float:
        return math.nan if x[0] < -0.5 else float(np.sum((x - 1) ** 2))

    result = minimize(f, x0=[0.0, 0.0])
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    "simplex",
    [
        np.zeros((2, 2)),  # wrong shape
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),  # collinear
    ],
)
def test_invalid_simplex_is_rejected(simplex: np.ndarray) -> None:
    with pytest.raises(ValueError, match="simplex"):
        minimize(FUNCTIONS["sphere"].func, initial_simplex=simplex)


@pytest.mark.parametrize(
    "kwargs",
    [{"reflection": 0}, {"expansion": 0.9}, {"contraction": 1.0}, {"shrink": 0.0}],
)
def test_invalid_coefficients_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="coefficient"):
        Coefficients(**kwargs)


def test_get_problem_validation() -> None:
    assert get_problem("rosenbrock").slug == "rosenbrock-2d"
    assert get_problem("sphere").slug == "sphere-3d"
    assert get_problem("rosenbrock", 5).initial_simplex.shape == (6, 5)
    with pytest.raises(ValueError, match="only defined"):
        get_problem("booth", 3)
    with pytest.raises(ValueError, match="unknown function"):
        get_problem("ackley")
