"""Nelder-Mead downhill simplex method.

The implementation follows the formulation in Lagarias, Reeds, Wright & Wright,
"Convergence Properties of the Nelder-Mead Simplex Method in Low Dimensions"
(SIAM J. Optim., 1998), which is also the variant used by
``scipy.optimize.minimize(method="Nelder-Mead")``:

* vertices are kept sorted so that ``f(x_1) <= ... <= f(x_{n+1})``;
* the worst vertex is reflected through the centroid of the other ``n``;
* expansion is greedy-minimization (the expanded point is kept only if it beats
  the reflected one);
* there are two contractions: *outside* (reflected point between the second
  worst and the worst) and *inside* (reflected point no better than the worst);
* if a contraction fails, the whole simplex shrinks towards the best vertex.

Termination uses the same pair of absolute tolerances as SciPy: the simplex must
be small in both the domain (``xatol``) and the image (``fatol``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Coefficients",
    "Result",
    "Step",
    "default_simplex",
    "minimize",
    "simplex_volume",
]

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
Objective = Callable[[FloatArray], float]


class Step(StrEnum):
    """Operation that produced a simplex from the previous one."""

    INITIAL = "initial"
    REFLECT = "reflect"
    EXPAND = "expand"
    CONTRACT_OUTSIDE = "contract_outside"
    CONTRACT_INSIDE = "contract_inside"
    SHRINK = "shrink"


@dataclass(frozen=True, slots=True)
class Coefficients:
    """Nelder-Mead coefficients (standard values by default).

    Attributes:
        reflection: ``alpha > 0``.
        expansion: ``gamma > 1`` and ``gamma > alpha``.
        contraction: ``0 < rho < 1`` (used for both outside and inside contraction).
        shrink: ``0 < sigma < 1``.
    """

    reflection: float = 1.0
    expansion: float = 2.0
    contraction: float = 0.5
    shrink: float = 0.5

    def __post_init__(self) -> None:
        if self.reflection <= 0:
            raise ValueError("reflection coefficient must be > 0")
        if self.expansion <= max(1.0, self.reflection):
            raise ValueError("expansion coefficient must be > 1 and > reflection")
        if not 0 < self.contraction < 1:
            raise ValueError("contraction coefficient must be in (0, 1)")
        if not 0 < self.shrink < 1:
            raise ValueError("shrink coefficient must be in (0, 1)")


@dataclass(slots=True)
class Result:
    """Outcome of a Nelder-Mead run.

    ``simplices[k]``/``fvalues[k]`` hold the sorted simplex after ``k`` iterations
    and ``steps[k]`` the operation that produced it (``steps[0]`` is ``INITIAL``).
    """

    x: FloatArray
    fun: float
    nit: int
    nfev: int
    converged: bool
    message: str
    simplices: list[FloatArray] = field(repr=False)
    fvalues: list[FloatArray] = field(repr=False)
    steps: list[Step] = field(repr=False)
    nfev_history: list[int] = field(repr=False)


def default_simplex(x0: ArrayLike, nonzdelt: float = 0.05, zdelt: float = 0.00025) -> FloatArray:
    """Build the initial simplex used by SciPy: perturb each coordinate of ``x0`` by 5 %."""
    x0 = np.asarray(x0, dtype=float).ravel()
    sim = np.tile(x0, (x0.size + 1, 1))
    for k in range(x0.size):
        sim[k + 1, k] = (1 + nonzdelt) * x0[k] if x0[k] != 0 else zdelt
    return sim


def simplex_volume(simplex: ArrayLike) -> float:
    """Volume of an ``n``-simplex given as an ``(n + 1, n)`` array of vertices."""
    sim = np.asarray(simplex, dtype=float)
    n = sim.shape[1]
    return abs(float(np.linalg.det(sim[1:] - sim[0]))) / math.factorial(n)


def _validate_simplex(simplex: FloatArray) -> None:
    if simplex.ndim != 2 or simplex.shape[0] != simplex.shape[1] + 1:
        raise ValueError(f"initial simplex must have shape (n + 1, n), got {simplex.shape}")
    if np.linalg.matrix_rank(simplex[1:] - simplex[0]) < simplex.shape[1]:
        raise ValueError("initial simplex is degenerate (vertices are affinely dependent)")


def minimize(
    func: Objective,
    x0: ArrayLike | None = None,
    *,
    initial_simplex: ArrayLike | None = None,
    coefficients: Coefficients | None = None,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    max_iter: int | None = None,
    max_fev: int | None = None,
) -> Result:
    """Minimize ``func`` with the Nelder-Mead simplex method.

    Args:
        func: Objective ``R^n -> R``. ``nan`` values are treated as ``+inf``.
        x0: Starting point; used to build a default simplex when
            ``initial_simplex`` is not given.
        initial_simplex: ``(n + 1, n)`` array of vertices.
        coefficients: Reflection/expansion/contraction/shrink coefficients
            (standard ``1, 2, 0.5, 0.5`` when omitted).
        xatol: Stop when ``max |x_i - x_1| <= xatol`` over all vertices...
        fatol: ...and ``max |f(x_i) - f(x_1)| <= fatol``.
        max_iter: Iteration budget (default ``200 * n``, as in SciPy).
        max_fev: Function-evaluation budget (default ``200 * n``, as in SciPy).

    Returns:
        A :class:`Result` with the best vertex and the full simplex history.
    """
    if initial_simplex is None:
        if x0 is None:
            raise ValueError("either x0 or initial_simplex is required")
        sim = default_simplex(x0)
    else:
        sim = np.array(initial_simplex, dtype=float)
    _validate_simplex(sim)

    n = sim.shape[1]
    max_iter = 200 * n if max_iter is None else max_iter
    max_fev = 200 * n if max_fev is None else max_fev
    coefficients = coefficients or Coefficients()
    alpha, gamma = coefficients.reflection, coefficients.expansion
    rho, sigma = coefficients.contraction, coefficients.shrink

    nfev = 0

    def f(x: FloatArray) -> float:
        nonlocal nfev
        nfev += 1
        value = float(func(x))
        return math.inf if math.isnan(value) else value

    fsim = np.array([f(v) for v in sim])

    def sort() -> None:
        nonlocal sim, fsim
        order = np.argsort(fsim, kind="stable")
        sim, fsim = sim[order], fsim[order]

    sort()
    simplices, fvalues, steps, nfevs = [sim.copy()], [fsim.copy()], [Step.INITIAL], [nfev]

    nit = 0
    converged = False
    while True:
        if (
            np.max(np.abs(sim[1:] - sim[0])) <= xatol
            and np.max(np.abs(fsim[1:] - fsim[0])) <= fatol
        ):
            converged = True
            message = "Converged: simplex size and f-spread below tolerance."
            break
        if nit >= max_iter:
            message = "Maximum number of iterations reached."
            break
        if nfev >= max_fev:
            message = "Maximum number of function evaluations reached."
            break

        centroid = sim[:-1].mean(axis=0)
        worst = sim[-1]
        xr = centroid + alpha * (centroid - worst)
        fr = f(xr)
        shrink = False

        if fr < fsim[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                sim[-1], fsim[-1], step = xe, fe, Step.EXPAND
            else:
                sim[-1], fsim[-1], step = xr, fr, Step.REFLECT
        elif fr < fsim[-2]:
            sim[-1], fsim[-1], step = xr, fr, Step.REFLECT
        elif fr < fsim[-1]:
            xc = centroid + rho * (xr - centroid)
            fc = f(xc)
            if fc <= fr:
                sim[-1], fsim[-1], step = xc, fc, Step.CONTRACT_OUTSIDE
            else:
                shrink = True
        else:
            xcc = centroid + rho * (worst - centroid)
            fcc = f(xcc)
            if fcc < fsim[-1]:
                sim[-1], fsim[-1], step = xcc, fcc, Step.CONTRACT_INSIDE
            else:
                shrink = True

        if shrink:
            sim[1:] = sim[0] + sigma * (sim[1:] - sim[0])
            fsim[1:] = [f(v) for v in sim[1:]]
            step = Step.SHRINK

        sort()
        nit += 1
        simplices.append(sim.copy())
        fvalues.append(fsim.copy())
        steps.append(step)
        nfevs.append(nfev)
        logger.debug("iter %d %-16s f_best=%.6e nfev=%d", nit, step, fsim[0], nfev)

    logger.debug("%s nit=%d nfev=%d f=%.6e", message, nit, nfev, fsim[0])
    return Result(
        x=sim[0].copy(),
        fun=float(fsim[0]),
        nit=nit,
        nfev=nfev,
        converged=converged,
        message=message,
        simplices=simplices,
        fvalues=fvalues,
        steps=steps,
        nfev_history=nfevs,
    )
