"""Gradient-based line-search methods: Steepest Descent, Newton and DFP (quasi-Newton).

All methods share one loop: pick a search direction ``p_k``, fall back to ``-g_k`` if
``p_k`` is not a descent direction, choose a step (Armijo backtracking; strong Wolfe for
DFP) and stop when ``||g_k|| <= tol``. Exact derivatives are used when supplied, central
finite differences otherwise (see :class:`Objective`).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DFP",
    "METHODS",
    "BaseOptimizer",
    "Newton",
    "Objective",
    "OptimizeResult",
    "SteepestDescent",
    "minimize",
]

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
#: A line-search outcome: step length, f at the new point, gradient there (if computed).
Step = tuple[float, float, FloatArray | None]


class Objective:
    """Objective with optional exact derivatives; counts every evaluation.

    Missing derivatives are approximated by central finite differences: the gradient from
    ``f`` (``2n`` calls, counted in ``nfev``), the Hessian from the gradient when it is
    available (``2n`` gradient calls) and from ``f`` otherwise (4-point stencil).
    """

    def __init__(
        self,
        fun: Callable[[FloatArray], float],
        grad: Callable[[FloatArray], FloatArray] | None = None,
        hess: Callable[[FloatArray], FloatArray] | None = None,
        grad_step: float = 1e-8,
        hess_step: float = 1e-5,
    ) -> None:
        self.fun = fun
        self._grad = grad
        self._hess = hess
        self.grad_step = grad_step
        self.hess_step = hess_step
        self.nfev = self.ngev = self.nhev = 0

    def __call__(self, x: FloatArray) -> float:
        self.nfev += 1
        return float(self.fun(x))

    def grad(self, x: FloatArray) -> FloatArray:
        self.ngev += 1
        if self._grad is not None:
            return np.asarray(self._grad(x), dtype=float)
        h = self.grad_step
        return np.array([(self(x + e) - self(x - e)) / (2 * h) for e in h * np.eye(x.size)])

    def hessian(self, x: FloatArray) -> FloatArray:
        self.nhev += 1
        if self._hess is not None:
            return np.asarray(self._hess(x), dtype=float)
        h, n = self.hess_step, x.size
        e = h * np.eye(n)
        if self._grad is not None:
            H = np.column_stack([(self.grad(x + ej) - self.grad(x - ej)) / (2 * h) for ej in e])
            return (H + H.T) / 2
        H = np.empty((n, n))
        for i in range(n):
            for j in range(i, n):
                H[i, j] = H[j, i] = (
                    self(x + e[i] + e[j])
                    - self(x + e[i] - e[j])
                    - self(x - e[i] + e[j])
                    + self(x - e[i] - e[j])
                ) / (4 * h * h)
        return H


@dataclass(slots=True)
class OptimizeResult:
    x: FloatArray
    fun: float
    nit: int
    nfev: int
    ngev: int
    nhev: int
    converged: bool
    message: str
    history: list[FloatArray]
    grad_norms: list[float]


class BaseOptimizer(ABC):
    """Shared line-search loop; subclasses only provide the search direction."""

    #: Armijo sufficient-decrease constant and backtracking factor.
    c1 = 1e-4
    backtrack = 0.5
    min_step = 1e-10

    def __init__(
        self,
        fun: Objective,
        x0: ArrayLike,
        tol: float = 1e-6,
        max_iter: int = 200,
        stop_when: Callable[[FloatArray], bool] | None = None,
    ) -> None:
        self.fun = fun
        self.x = np.array(x0, dtype=float)
        self.tol = tol
        self.max_iter = max_iter
        self.stop_when = stop_when  # optional extra stopping test, checked after each step

    @abstractmethod
    def direction(self, x: FloatArray, g: FloatArray) -> FloatArray:
        """Search direction at ``x`` with gradient ``g``."""

    def reset(self) -> None:  # noqa: B027 (optional hook, no-op by default)
        """Called when the direction had to fall back to ``-g``."""

    def update(self, s: FloatArray, y: FloatArray) -> None:  # noqa: B027 (optional hook)
        """Called after each step with ``s = x_{k+1} - x_k`` and ``y = g_{k+1} - g_k``."""

    def line_search(self, x: FloatArray, fx: float, g: FloatArray, p: FloatArray) -> Step:
        """Backtrack until ``f(x + a p) <= f(x) + c1 a g^T p`` (Armijo).

        Returns ``(a, f(x + a p), None)``, or ``a = 0`` when no acceptable step larger than
        ``min_step`` exists.
        """
        alpha, slope = 1.0, float(g @ p)
        while alpha >= self.min_step:
            f_new = self.fun(x + alpha * p)
            if f_new <= fx + self.c1 * alpha * slope:
                return alpha, f_new, None
            alpha *= self.backtrack
        return 0.0, fx, None

    def optimize(self) -> OptimizeResult:
        x = self.x
        fx = self.fun(x)
        g = self.fun.grad(x)
        history, grad_norms = [x.copy()], [float(np.linalg.norm(g))]
        converged, message = False, "Maximum number of iterations reached."
        nit = 0
        while True:
            if grad_norms[-1] <= self.tol:
                converged, message = True, "Gradient norm below tolerance."
                break
            if self.stop_when is not None and nit > 0 and self.stop_when(x):
                converged, message = True, "Stopping condition met."
                break
            if nit >= self.max_iter:
                break
            p = self.direction(x, g)
            if not np.all(np.isfinite(p)) or p @ g >= 0:
                p = -g
                self.reset()
            alpha, f_new, g_new = self.line_search(x, fx, g, p)
            if alpha == 0.0:
                message = "Line search could not decrease f (precision limit of the gradient)."
                break
            x_new = x + alpha * p
            if g_new is None:
                g_new = self.fun.grad(x_new)
            self.update(x_new - x, g_new - g)
            x, g, fx = x_new, g_new, f_new
            nit += 1
            history.append(x.copy())
            grad_norms.append(float(np.linalg.norm(g)))
        self.x = x
        logger.debug("%s: %s nit=%d nfev=%d", type(self).__name__, message, nit, self.fun.nfev)
        return OptimizeResult(
            x=x,
            fun=fx,
            nit=nit,
            nfev=self.fun.nfev,
            ngev=self.fun.ngev,
            nhev=self.fun.nhev,
            converged=converged,
            message=message,
            history=history,
            grad_norms=grad_norms,
        )


class SteepestDescent(BaseOptimizer):
    """``p_k = -g_k``."""

    def direction(self, x: FloatArray, g: FloatArray) -> FloatArray:
        return -g


class Newton(BaseOptimizer):
    """``p_k = -H_k^{-1} g_k`` with a finite-difference Hessian."""

    max_step_norm = 100.0

    def direction(self, x: FloatArray, g: FloatArray) -> FloatArray:
        try:
            p = np.linalg.solve(self.fun.hessian(x), -g)
        except np.linalg.LinAlgError:
            return -g
        # A huge Newton step means an ill-conditioned Hessian; use the gradient instead.
        return -g if np.linalg.norm(p) > self.max_step_norm else p


class DFP(BaseOptimizer):
    """Davidon-Fletcher-Powell: ``p_k = -B_k g_k`` with ``B_k`` approximating ``H^{-1}``.

    ``B_{k+1} = B_k + s s^T / (s^T y) - B_k y y^T B_k / (y^T B_k y)``, applied only when
    the curvature condition ``s^T y > 0`` holds, which keeps ``B`` positive definite.

    Quasi-Newton updates need steps that satisfy the Wolfe conditions, so DFP uses a
    strong-Wolfe line search instead of plain Armijo backtracking. Armijo alone accepts
    steps that are far too short along curved valleys: ``s^T y`` is then tiny or negative,
    updates are skipped or badly scaled, and the method crawls (on Rosenbrock it ran out of
    500 iterations from most starting points within 1e-9 of ``(-1.2, 1)``).
    """

    c2 = 0.1  # strong-Wolfe curvature constant: a fairly accurate line search, as DFP needs
    max_step = 1e3
    max_trials = 40

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.B = np.eye(self.x.size)

    def direction(self, x: FloatArray, g: FloatArray) -> FloatArray:
        return -self.B @ g

    def reset(self) -> None:
        self.B = np.eye(self.x.size)

    def line_search(self, x: FloatArray, fx: float, g: FloatArray, p: FloatArray) -> Step:
        """Strong Wolfe line search (Nocedal & Wright, Algorithms 3.5 and 3.6, bisection zoom).

        Finds ``a`` with ``f(x + a p) <= f(x) + c1 a g^T p`` and
        ``|grad f(x + a p)^T p| <= c2 |g^T p|``. If the trial budget runs out, it returns the
        best step that satisfies the sufficient-decrease condition, or ``a = 0`` if none does.
        """
        slope0 = float(g @ p)
        trials = 0

        def armijo_fails(a: float, f_a: float) -> bool:
            return not math.isfinite(f_a) or f_a > fx + self.c1 * a * slope0

        def zoom(lo: float, f_lo: float, g_lo: FloatArray | None, hi: float) -> Step:
            nonlocal trials
            while trials < self.max_trials and abs(hi - lo) > self.min_step:
                trials += 1
                a = 0.5 * (lo + hi)
                f_a = self.fun(x + a * p)
                if armijo_fails(a, f_a) or f_a >= f_lo:
                    hi = a
                    continue
                g_a = self.fun.grad(x + a * p)
                d_a = float(g_a @ p)
                if abs(d_a) <= -self.c2 * slope0:
                    return a, f_a, g_a
                if d_a * (hi - lo) >= 0:
                    hi = lo
                lo, f_lo, g_lo = a, f_a, g_a
            return (lo, f_lo, g_lo) if lo > 0 else (0.0, fx, None)

        a_prev, f_prev, g_prev = 0.0, fx, None
        a = 1.0
        while trials < self.max_trials:
            trials += 1
            f_a = self.fun(x + a * p)
            if armijo_fails(a, f_a) or (a_prev > 0 and f_a >= f_prev):
                return zoom(a_prev, f_prev, g_prev, a)
            g_a = self.fun.grad(x + a * p)
            d_a = float(g_a @ p)
            if abs(d_a) <= -self.c2 * slope0:
                return a, f_a, g_a
            if d_a >= 0:
                return zoom(a, f_a, g_a, a_prev)
            a_prev, f_prev, g_prev = a, f_a, g_a
            if a >= self.max_step:
                break
            a = min(2 * a, self.max_step)
        return (a_prev, f_prev, g_prev) if a_prev > 0 else (0.0, fx, None)

    def update(self, s: FloatArray, y: FloatArray) -> None:
        sy = float(s @ y)
        By = self.B @ y
        yBy = float(y @ By)
        if sy <= 1e-12 * np.linalg.norm(s) * np.linalg.norm(y) or yBy <= 0 or not math.isfinite(sy):
            return
        self.B += np.outer(s, s) / sy - np.outer(By, By) / yBy


METHODS: dict[str, type[BaseOptimizer]] = {
    "SD": SteepestDescent,
    "Newton": Newton,
    "DFP": DFP,
}


def minimize(
    fun: Callable[[FloatArray], float],
    x0: ArrayLike,
    method: str = "DFP",
    *,
    grad: Callable[[FloatArray], FloatArray] | None = None,
    hess: Callable[[FloatArray], FloatArray] | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> OptimizeResult:
    """Minimize ``fun`` from ``x0`` with one of :data:`METHODS` (``SD``, ``Newton``, ``DFP``)."""
    try:
        cls = METHODS[method]
    except KeyError:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}") from None
    return cls(Objective(fun, grad, hess), x0, tol=tol, max_iter=max_iter).optimize()
