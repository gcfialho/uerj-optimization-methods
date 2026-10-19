"""Benchmark objective functions and the bundled demo problems.

Every function is vectorized over the last axis, so ``f(X)`` with ``X`` of shape
``(..., n)`` returns an array of shape ``(...)``. This lets the plotting code
evaluate a whole grid in one call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nelder_mead.optimizer import FloatArray, default_simplex

__all__ = ["FUNCTIONS", "PROBLEMS", "Benchmark", "Problem", "get_problem"]

SHIFT = np.array([1.0, -2.0, 0.5])


def sphere(x: ArrayLike) -> NDArray[np.float64] | float:
    """``sum x_i^2``; minimum 0 at the origin."""
    x = np.asarray(x, dtype=float)
    return np.sum(x**2, axis=-1)


def shifted_sphere(x: ArrayLike) -> NDArray[np.float64] | float:
    """``sum (x_i - c_i)^2`` with ``c = (1, -2, 0.5, 1, -2, ...)``; minimum 0 at ``c``."""
    x = np.asarray(x, dtype=float)
    return np.sum((x - np.resize(SHIFT, x.shape[-1])) ** 2, axis=-1)


def rosenbrock(x: ArrayLike) -> NDArray[np.float64] | float:
    """``sum 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2``; minimum 0 at ``(1, ..., 1)``."""
    x = np.asarray(x, dtype=float)
    head, tail = x[..., :-1], x[..., 1:]
    return np.sum(100.0 * (tail - head**2) ** 2 + (1.0 - head) ** 2, axis=-1)


def booth(x: ArrayLike) -> NDArray[np.float64] | float:
    """``(x + 2y - 7)^2 + (2x + y - 5)^2``; minimum 0 at ``(1, 3)``."""
    x = np.asarray(x, dtype=float)
    u, v = x[..., 0], x[..., 1]
    return (u + 2 * v - 7) ** 2 + (2 * u + v - 5) ** 2


def himmelblau(x: ArrayLike) -> NDArray[np.float64] | float:
    """``(x^2 + y - 11)^2 + (x + y^2 - 7)^2``; four global minima with value 0."""
    x = np.asarray(x, dtype=float)
    u, v = x[..., 0], x[..., 1]
    return (u**2 + v - 11) ** 2 + (u + v**2 - 7) ** 2


HIMMELBLAU_MINIMA = np.array(
    [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ]
)


@dataclass(frozen=True, slots=True)
class Benchmark:
    """A benchmark function with its known global minimizers."""

    name: str
    func: Callable[[ArrayLike], NDArray[np.float64] | float]
    minimizers: Callable[[int], FloatArray]
    default_start: Callable[[int], FloatArray]
    dim: int | None = None  # fixed dimension, or None for any n >= 2

    def error(self, x: ArrayLike) -> float:
        """Euclidean distance from ``x`` to the nearest known global minimizer."""
        x = np.asarray(x, dtype=float)
        return float(np.min(np.linalg.norm(self.minimizers(x.size) - x, axis=1)))


FUNCTIONS: dict[str, Benchmark] = {
    f.name: f
    for f in (
        Benchmark(
            "sphere",
            sphere,
            minimizers=lambda n: np.zeros((1, n)),
            default_start=lambda n: np.full(n, 3.0),
        ),
        Benchmark(
            "shifted-sphere",
            shifted_sphere,
            minimizers=lambda n: np.resize(SHIFT, (1, n)),
            default_start=lambda n: np.full(n, -1.0),
        ),
        Benchmark(
            "rosenbrock",
            rosenbrock,
            minimizers=lambda n: np.ones((1, n)),
            default_start=lambda n: np.resize([-1.2, 1.0], n),
        ),
        Benchmark(
            "booth",
            booth,
            minimizers=lambda n: np.array([[1.0, 3.0]]),
            default_start=lambda n: np.zeros(2),
            dim=2,
        ),
        Benchmark(
            "himmelblau",
            himmelblau,
            minimizers=lambda n: HIMMELBLAU_MINIMA,
            default_start=lambda n: np.zeros(2),
            dim=2,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class Problem:
    """A function, a dimension and an initial simplex (plus plot limits for 2-D)."""

    function: Benchmark
    initial_simplex: FloatArray
    bounds: tuple[tuple[float, float], ...] | None = None

    @property
    def dim(self) -> int:
        return self.initial_simplex.shape[1]

    @property
    def slug(self) -> str:
        return f"{self.function.name}-{self.dim}d"


def _problem(
    name: str, simplex: list[list[float]], bounds: tuple[tuple[float, float], ...] | None = None
) -> Problem:
    return Problem(FUNCTIONS[name], np.array(simplex, dtype=float), bounds)


# The demo problems shipped with the original coursework (same initial simplices).
PROBLEMS: dict[str, Problem] = {
    p.slug: p
    for p in (
        _problem("rosenbrock", [[-1.2, 1.0], [0.0, 1.2], [-1.0, 0.8]], ((-2.0, 2.0), (-1.0, 3.0))),
        _problem("booth", [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]], ((-1.0, 5.0), (-1.0, 5.0))),
        _problem(
            "himmelblau", [[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]], ((-5.0, 5.0), (-5.0, 5.0))
        ),
        _problem(
            "sphere",
            [[5.0, -3.0, 2.0], [5.05, -3.0, 2.0], [5.0, -2.95, 2.0], [5.0, -3.0, 2.05]],
        ),
        _problem(
            "shifted-sphere",
            [[-1.0, -1.0, -1.0], [-0.9, -1.0, -1.0], [-1.0, -0.9, -1.0], [-1.0, -1.0, -0.9]],
        ),
    )
}


def get_problem(function: str, dim: int | None = None) -> Problem:
    """Return the bundled demo for ``(function, dim)`` or build one from defaults.

    Without a bundled demo, the initial simplex is SciPy's default simplex around
    the function's default starting point.
    """
    try:
        benchmark = FUNCTIONS[function]
    except KeyError:
        raise ValueError(
            f"unknown function {function!r}; choose from {sorted(FUNCTIONS)}"
        ) from None
    if benchmark.dim is not None:
        if dim not in (None, benchmark.dim):
            raise ValueError(f"{function} is only defined for dim={benchmark.dim}")
        dim = benchmark.dim
    if dim is None:
        dim = next((p.dim for p in PROBLEMS.values() if p.function is benchmark), 2)
    if dim < 2:
        raise ValueError("dim must be >= 2")
    slug = f"{function}-{dim}d"
    if slug in PROBLEMS:
        return PROBLEMS[slug]
    return Problem(benchmark, default_simplex(benchmark.default_start(dim)))
