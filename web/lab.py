"""Bridge between the browser lab and the two packages.

The lab runs this module in the browser with Pyodide, next to unmodified copies of
``nelder_mead`` and ``tank_optimizer``, so the page executes the same tested code as the
CLIs. Every function takes and returns plain JSON-ready values.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from nelder_mead import FUNCTIONS, PROBLEMS, Coefficients, Step, default_simplex, minimize
from tank_optimizer.service import run_comparison, run_optimization

__all__ = ["call", "nm_functions", "nm_run", "tank_compare", "tank_run"]

# 2-D functions offered in the lab, with plot limits. Bundled demos keep their original
# coursework simplex; the sphere uses SciPy's default simplex around its default start.
NM_FUNCTIONS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "rosenbrock": ((-2.0, 2.0), (-1.0, 3.0)),
    "booth": ((-1.0, 5.0), (-1.0, 5.0)),
    "himmelblau": ((-5.0, 5.0), (-5.0, 5.0)),
    "sphere": ((-4.0, 4.0), (-4.0, 4.0)),
}
GRID = 90


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def tank_run(params: dict[str, Any]) -> dict[str, Any]:
    """One method (``SD``, ``Newton`` or ``DFP``); same payload as ``POST /optimize``."""
    return run_optimization(params)


def tank_compare(params: dict[str, Any]) -> dict[str, Any]:
    """Every method from the same start; same payload as ``POST /compare``."""
    return run_comparison(params)


def nm_functions() -> dict[str, Any]:
    """Contour grids, default simplices and known minimizers for the 2-D functions."""
    out: dict[str, Any] = {}
    for name, ((x0, x1), (y0, y1)) in NM_FUNCTIONS.items():
        bench = FUNCTIONS[name]
        xs = np.linspace(x0, x1, GRID)
        ys = np.linspace(y0, y1, GRID)
        X, Y = np.meshgrid(xs, ys)
        Z = np.asarray(bench.func(np.stack([X, Y], axis=-1)), dtype=float)
        demo = PROBLEMS.get(f"{name}-2d")
        simplex = demo.initial_simplex if demo else default_simplex(bench.default_start(2))
        out[name] = {
            "x": xs.tolist(),
            "y": ys.tolist(),
            "z": Z.tolist(),
            "simplex": simplex.tolist(),
            "minimizers": bench.minimizers(2).tolist(),
            "bounds": [[x0, x1], [y0, y1]],
        }
    return out


def nm_run(
    function: str,
    start: list[float] | None = None,
    coefficients: dict[str, float] | None = None,
    max_iter: int = 400,
) -> dict[str, Any]:
    """Run Nelder-Mead on a 2-D function and return the full simplex history.

    With ``start`` the initial simplex is SciPy's default simplex around it; otherwise the
    function's default (coursework) simplex is used. Invalid coefficients raise
    ``ValueError`` with the package's message.
    """
    if function not in NM_FUNCTIONS:
        raise ValueError(f"unknown function {function!r}")
    bench = FUNCTIONS[function]
    demo = PROBLEMS.get(f"{function}-2d")
    if start is not None:
        simplex = default_simplex(np.asarray(start, dtype=float))
    elif demo is not None:
        simplex = demo.initial_simplex
    else:
        simplex = default_simplex(bench.default_start(2))
    coeffs = Coefficients(**coefficients) if coefficients else Coefficients()
    result = minimize(
        bench.func,
        initial_simplex=simplex,
        coefficients=coeffs,
        max_iter=max_iter,
        max_fev=max_iter * 4,
    )
    counts = {step.value: 0 for step in Step if step is not Step.INITIAL}
    for step in result.steps[1:]:
        counts[step.value] += 1
    return {
        "x": result.x.tolist(),
        "fun": float(result.fun),
        "nit": result.nit,
        "nfev": result.nfev,
        "converged": result.converged,
        "message": result.message,
        "distance": bench.error(result.x),
        "simplices": [s.tolist() for s in result.simplices],
        "best": [_finite(float(f[0])) for f in result.fvalues],
        "steps": [s.value for s in result.steps],
        "nfev_history": list(result.nfev_history),
        "counts": counts,
    }


def call(name: str, payload: str) -> str:
    """Entry point for JavaScript: JSON in, JSON out, errors as ``{"error": message}``."""
    import json

    functions: dict[str, Callable[..., object]] = {"nm_functions": nm_functions, "nm_run": nm_run}
    functions |= {"tank_run": tank_run, "tank_compare": tank_compare}
    try:
        args = json.loads(payload) if payload else {}
        result = functions[name](args) if name.startswith("tank") else functions[name](**args)
    except (ValueError, TypeError, KeyError) as exc:
        import traceback

        return json.dumps({"error": str(exc), "trace": traceback.format_exc()})
    return json.dumps(result, allow_nan=False)
