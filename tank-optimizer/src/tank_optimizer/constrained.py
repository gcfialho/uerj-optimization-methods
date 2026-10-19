"""Constrained solve by a sequence of unconstrained minimizations (SUMT).

1. **Exterior penalty** (only while the start is not strictly feasible): minimize
   ``C + r * sum(max(0, g + margin)^2)`` for ``r = 1e3, 1e4, ...`` until the iterate is
   strictly feasible.
2. **Interior barrier**: minimize ``C - mu * sum(log(-g))`` for ``mu = 10, 1, ..., 1e-3``,
   warm-starting each subproblem from the previous solution. The barrier minimizer lies on
   the central path, ``O(mu)`` away from the constrained optimum in cost.

Every subproblem is solved by the selected line-search method (SD, Newton or DFP) with the
exact derivatives from :mod:`tank_optimizer.objective`; the iteration budget is shared.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from tank_optimizer.objective import (
    SPEC,
    Subproblem,
    TankSpec,
    barrier_problem,
    constraints,
    cost,
    penalty_problem,
)
from tank_optimizer.optimizers import METHODS, FloatArray, Objective

__all__ = ["ConstrainedResult", "StageResult", "solve"]

Stop = Callable[[FloatArray], bool]

PENALTY_START, PENALTY_GROWTH, PENALTY_MAX = 1e5, 10.0, 1e9
PENALTY_MARGIN = 1e-3  # constraints are shifted inwards by this much during phase 1
BARRIER_START, BARRIER_DECAY, BARRIER_MIN = 10.0, 0.1, 1e-2


@dataclass(slots=True)
class StageResult:
    kind: str
    weight: float
    nit: int
    converged: bool
    message: str


@dataclass(slots=True)
class ConstrainedResult:
    x: FloatArray
    cost: float
    nit: int
    nfev: int
    ngev: int
    nhev: int
    feasible: bool
    converged: bool
    message: str
    history: list[FloatArray]
    grad_norms: list[float]
    stages: list[StageResult] = field(default_factory=list)


def _strictly_feasible(x: FloatArray, spec: TankSpec) -> bool:
    return bool(x[0] > 0 and x[1] > 0 and np.all(constraints(x[0], x[1], spec) < 0))


def solve(
    x0: ArrayLike,
    method: str = "Newton",
    tol: float = 1e-6,
    max_iter: int = 200,
    spec: TankSpec = SPEC,
) -> ConstrainedResult:
    """Minimize the tank cost subject to ``g(x) <= 0`` starting from ``x0 = (D0, L0)``.

    ``tol`` is the gradient-norm tolerance of the last barrier subproblem; earlier
    subproblems only need to be solved to ``max(tol, 1e-2)``.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    x = np.array(x0, dtype=float)
    out = ConstrainedResult(
        x=x, cost=math.nan, nit=0, nfev=0, ngev=0, nhev=0, feasible=False, converged=False,
        message="", history=[x.copy()], grad_norms=[],
    )  # fmt: skip

    def run(problem: Subproblem, final: bool, stop_when: Stop | None = None) -> bool:
        """Solve one subproblem from the current ``x``; False once the budget is spent.

        Intermediate subproblems get a loose tolerance and at most a quarter of the budget,
        so slow methods still move along the penalty/barrier path.
        """
        nonlocal x
        budget = max_iter - out.nit
        if budget <= 0:
            return False
        stage_tol, cap = (tol, budget) if final else (max(tol, 1e-2), max(max_iter // 4, 1))
        fun = Objective(problem.fun, problem.grad, problem.hess)
        solver = METHODS[method](
            fun, x, tol=stage_tol, max_iter=min(cap, budget), stop_when=stop_when
        )
        res = solver.optimize()
        x = res.x
        out.nit += res.nit
        out.nfev, out.ngev, out.nhev = out.nfev + res.nfev, out.ngev + res.ngev, out.nhev + res.nhev
        out.history.extend(res.history[1:])
        # The first norm belongs to the start of this subproblem; keep it only for stage one.
        out.grad_norms.extend(res.grad_norms if not out.grad_norms else res.grad_norms[1:])
        out.stages.append(
            StageResult(problem.kind, problem.weight, res.nit, res.converged, res.message)
        )
        return True

    def feasible(z: FloatArray) -> bool:
        return _strictly_feasible(z, spec)

    r = PENALTY_START
    while not feasible(x) and r <= PENALTY_MAX:
        # Phase 1 only has to reach the interior: stop as soon as the iterate gets there.
        if not run(penalty_problem(r, PENALTY_MARGIN, spec), final=False, stop_when=feasible):
            break
        r *= PENALTY_GROWTH

    if feasible(x):
        mu = BARRIER_START
        while mu >= BARRIER_MIN * (1 - 1e-9):
            last = mu * BARRIER_DECAY < BARRIER_MIN * (1 - 1e-9)
            if not run(barrier_problem(mu, spec), final=last):
                break
            if last:
                out.converged = out.stages[-1].converged
            mu *= BARRIER_DECAY

    D, L = x
    out.x = x
    out.cost = float(cost(D, L, spec))
    out.feasible = bool(D > 0 and L > 0 and np.all(constraints(D, L, spec) <= 0))
    if not out.feasible:
        out.message = "No feasible design found within the iteration budget."
    elif out.converged:
        out.message = f"Converged: ||grad|| <= {tol:g} at barrier weight {BARRIER_MIN:g}."
    elif out.nit >= max_iter:
        out.message = (
            "Feasible; iteration budget spent before the last barrier subproblem converged."
        )
    else:
        out.message = f"Feasible; last subproblem stopped early: {out.stages[-1].message}"
    return out
