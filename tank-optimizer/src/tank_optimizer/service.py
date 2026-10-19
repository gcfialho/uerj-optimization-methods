"""Service layer: validates UI parameters, runs the optimizers and shapes the JSON payload."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tank_optimizer.constrained import ConstrainedResult, solve
from tank_optimizer.objective import SPEC, cost, volume
from tank_optimizer.optimizers import METHODS

__all__ = ["InvalidRequestError", "contour_grid", "run_comparison", "run_optimization"]

D_RANGE = (0.1, 1.2)
L_RANGE = (0.1, 2.2)
GRID_POINTS = 50


class InvalidRequestError(ValueError):
    """Raised for malformed or out-of-range request parameters."""


def _parse(params: dict[str, Any]) -> tuple[np.ndarray, float, int]:
    try:
        x0 = np.array([float(params["D0"]), float(params["L0"])])
        tol = float(params.get("tol", 1e-6))
        max_iter = int(params.get("max_iter", 200))
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidRequestError(f"invalid parameters: {exc}") from None
    if not np.all(np.isfinite(x0)) or np.any(x0 <= 0):
        raise InvalidRequestError("D0 and L0 must be positive numbers")
    if not (math.isfinite(tol) and tol > 0):
        raise InvalidRequestError("tol must be a positive number")
    if not 1 <= max_iter <= 10_000:
        raise InvalidRequestError("max_iter must be between 1 and 10000")
    return x0, tol, max_iter


def _finite(values: list[float]) -> list[float | None]:
    """JSON has no inf/nan; send ``null`` instead."""
    return [v if math.isfinite(v) else None for v in values]


def _summary(result: ConstrainedResult) -> dict[str, Any]:
    D, L = (float(v) for v in result.x)
    return {
        "history": np.asarray(result.history, dtype=float).tolist(),
        "errors": _finite(result.grad_norms),
        "final_D": D,
        "final_L": L,
        "final_cost": result.cost,
        "final_volume": float(volume(D, L)),
        "feasible": result.feasible,
        "iterations": result.nit,
        "num_eval": result.nfev,
        "num_grad": result.ngev,
        "num_hess": result.nhev,
        "converged": result.converged,
        "message": result.message,
        "stages": [
            {"kind": st.kind, "weight": st.weight, "iterations": st.nit} for st in result.stages
        ],
    }


def contour_grid() -> dict[str, Any]:
    """Cost surface on a regular grid plus the problem limits used to draw the feasible set."""
    d = np.linspace(*D_RANGE, GRID_POINTS)
    l_ = np.linspace(*L_RANGE, GRID_POINTS)
    D, L = np.meshgrid(d, l_)
    return {
        "d": d.tolist(),
        "l": l_.tolist(),
        "cost": cost(D, L).tolist(),
        "limits": {
            "v_min": SPEC.min_volume,
            "v_max": SPEC.max_volume,
            "d_max": SPEC.max_diameter,
            "l_max": SPEC.max_length,
        },
    }


def run_optimization(params: dict[str, Any]) -> dict[str, Any]:
    """Run the method named in ``params["method"]`` (``SD``, ``Newton`` or ``DFP``)."""
    method = str(params.get("method", "SD"))
    if method not in METHODS:
        raise InvalidRequestError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    x0, tol, max_iter = _parse(params)
    return {**_summary(solve(x0, method, tol, max_iter)), "contour": contour_grid()}


def run_comparison(params: dict[str, Any]) -> dict[str, Any]:
    """Run every method from the same starting point."""
    x0, tol, max_iter = _parse(params)
    results = {name: _summary(solve(x0, name, tol, max_iter)) for name in METHODS}
    return {"results": results, "contour": contour_grid()}
