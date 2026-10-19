"""Cylindrical tank design: cost model, constraints and the penalized objective.

The tank is a closed cylinder with inner diameter ``D`` and length ``L`` (metres),
rolled from a plate of thickness ``t`` and closed by two flat end plates. Its cost is
the material cost of the steel plus the cost of welding the two end plates:

    mass(D, L) = rho * [ pi L ((D/2 + t)^2 - (D/2)^2)   # cylindrical shell
                       + 2 pi (D/2 + t)^2 t ]           # two end plates
    weld(D)    = 4 pi (D + t)                           # 2 seams per end plate
    C(D, L)    = c_m * mass + c_w * weld

subject to a +/-10 % tolerance on the internal volume and maximum dimensions:

    g1 = 0.9 V0 - pi D^2 L / 4 <= 0
    g2 = pi D^2 L / 4 - 1.1 V0 <= 0
    g3 = L - L_max <= 0
    g4 = D - D_max <= 0

The constrained problem is solved as a sequence of unconstrained ones (SUMT, see
tank_optimizer.constrained): an exterior penalty while the design is infeasible, then an
interior log barrier with a decreasing weight.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SPEC",
    "Subproblem",
    "TankSpec",
    "barrier_problem",
    "constraints",
    "cost",
    "cost_gradient",
    "cost_hessian",
    "is_feasible",
    "mass",
    "penalty_problem",
    "volume",
    "weld_length",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TankSpec:
    """Problem data (SI units; costs in $)."""

    volume: float = 0.8  # V0, required internal volume [m^3]
    volume_tolerance: float = 0.1  # accepted relative deviation from V0
    thickness: float = 0.03  # t, plate thickness [m]
    density: float = 8000.0  # rho, steel density [kg/m^3]
    max_length: float = 2.0  # L_max [m]
    max_diameter: float = 1.0  # D_max [m]
    material_cost: float = 4.5  # c_m [$/kg]
    weld_cost: float = 20.0  # c_w [$/m]

    @property
    def min_volume(self) -> float:
        return (1 - self.volume_tolerance) * self.volume

    @property
    def max_volume(self) -> float:
        return (1 + self.volume_tolerance) * self.volume


SPEC = TankSpec()


def volume(D: ArrayLike, L: ArrayLike) -> FloatArray:
    """Internal volume ``pi D^2 L / 4`` [m^3]."""
    return np.pi * np.asarray(D) ** 2 * np.asarray(L) / 4


def mass(D: ArrayLike, L: ArrayLike, spec: TankSpec = SPEC) -> FloatArray:
    """Steel mass of the shell plus the two end plates [kg]."""
    D, L, t = np.asarray(D), np.asarray(L), spec.thickness
    outer_r2 = (D / 2 + t) ** 2
    shell = np.pi * L * (outer_r2 - (D / 2) ** 2)
    end_plates = 2 * np.pi * outer_r2 * t
    return spec.density * (shell + end_plates)


def weld_length(D: ArrayLike, spec: TankSpec = SPEC) -> FloatArray:
    """Total weld length: two circumferential seams at the mean diameter per end [m]."""
    return 4 * np.pi * (np.asarray(D) + spec.thickness)


def cost(D: ArrayLike, L: ArrayLike, spec: TankSpec = SPEC) -> FloatArray:
    """Manufacturing cost ``C(D, L)`` [$]. Vectorized over ``D`` and ``L``."""
    return spec.material_cost * mass(D, L, spec) + spec.weld_cost * weld_length(D, spec)


def constraints(D: ArrayLike, L: ArrayLike, spec: TankSpec = SPEC) -> FloatArray:
    """Constraint vector ``g(D, L)``; the design is feasible iff ``g <= 0``."""
    vol = volume(D, L)
    return np.array(
        [
            spec.min_volume - vol,
            vol - spec.max_volume,
            np.asarray(L) - spec.max_length,
            np.asarray(D) - spec.max_diameter,
        ]
    )


def is_feasible(D: float, L: float, spec: TankSpec = SPEC, tol: float = 0.0) -> bool:
    return bool(D > 0 and L > 0 and np.all(constraints(D, L, spec) <= tol))


# --- derivatives --------------------------------------------------------------------
#
# Every term is a polynomial (or the log of one), so the derivatives are exact. Finite
# differences are a poor fit here: near the barrier the curvature is ~1e9, and the original
# penalty (cost + 1e12 outside the feasible set) made central differences return a gradient
# of exactly zero, because the float spacing at 1e12 (~1e-4) exceeds the change of f.


def cost_gradient(D: float, L: float, spec: TankSpec = SPEC) -> FloatArray:
    a, t = spec.material_cost * spec.density, spec.thickness
    return np.array(
        [
            a * np.pi * t * (L + D + 2 * t) + 4 * np.pi * spec.weld_cost,
            a * np.pi * t * (D + t),
        ]
    )


def cost_hessian(spec: TankSpec = SPEC) -> FloatArray:
    """Constant: the cost is linear in ``L`` and quadratic in ``D``."""
    k = spec.material_cost * spec.density * np.pi * spec.thickness
    return np.array([[k, k], [k, 0.0]])


def _constraint_derivatives(D: float, L: float) -> tuple[FloatArray, FloatArray]:
    """Gradients (4 x 2) and Hessians (4 x 2 x 2) of ``g``."""
    dvol = np.array([np.pi * D * L / 2, np.pi * D**2 / 4])
    d2vol = np.array([[np.pi * L / 2, np.pi * D / 2], [np.pi * D / 2, 0.0]])
    zero = np.zeros((2, 2))
    return (
        np.array([-dvol, dvol, [0.0, 1.0], [1.0, 0.0]]),
        np.array([-d2vol, d2vol, zero, zero]),
    )


# --- unconstrained surrogates (SUMT) -------------------------------------------------


@dataclass(frozen=True, slots=True)
class Subproblem:
    """An unconstrained surrogate ``phi`` with its exact derivatives."""

    kind: str
    weight: float
    fun: Callable[[FloatArray], float]
    grad: Callable[[FloatArray], FloatArray]
    hess: Callable[[FloatArray], FloatArray]


def barrier_problem(mu: float, spec: TankSpec = SPEC) -> Subproblem:
    """Interior log barrier ``C(x) - mu * sum(log(-g_i(x)))`` (``+inf`` unless ``g(x) < 0``)."""

    def fun(x: FloatArray) -> float:
        D, L = x
        g = constraints(D, L, spec)
        if D <= 0 or L <= 0 or np.any(g >= 0):
            return math.inf
        return float(cost(D, L, spec)) - mu * float(np.sum(np.log(-g)))

    def grad(x: FloatArray) -> FloatArray:
        D, L = x
        dg, _ = _constraint_derivatives(D, L)
        return cost_gradient(D, L, spec) - mu * ((1 / constraints(D, L, spec)) @ dg)

    def hess(x: FloatArray) -> FloatArray:
        D, L = x
        g = constraints(D, L, spec)
        dg, d2g = _constraint_derivatives(D, L)
        outer = np.einsum("ki,kj->kij", dg, dg)
        terms = outer / (g**2)[:, None, None] - d2g / g[:, None, None]
        return cost_hessian(spec) + mu * terms.sum(axis=0)

    return Subproblem("barrier", mu, fun, grad, hess)


def penalty_problem(r: float, margin: float, spec: TankSpec = SPEC) -> Subproblem:
    """Exterior penalty ``C(x) + r * sum(max(0, g_i(x) + margin)^2)``.

    The margin shifts the constraints inwards, so the penalty minimizer (which violates the
    shifted constraints by ``O(1/r)``) becomes strictly feasible once ``r`` is large enough.
    """

    def fun(x: FloatArray) -> float:
        D, L = x
        if D <= 0 or L <= 0:
            return math.inf
        h = np.maximum(constraints(D, L, spec) + margin, 0)
        return float(cost(D, L, spec)) + r * float(h @ h)

    def grad(x: FloatArray) -> FloatArray:
        D, L = x
        h = np.maximum(constraints(D, L, spec) + margin, 0)
        dg, _ = _constraint_derivatives(D, L)
        return cost_gradient(D, L, spec) + 2 * r * (h @ dg)

    def hess(x: FloatArray) -> FloatArray:
        D, L = x
        h = np.maximum(constraints(D, L, spec) + margin, 0)
        dg, d2g = _constraint_derivatives(D, L)
        active = h > 0
        outer = np.einsum("ki,kj->kij", dg[active], dg[active])
        terms = outer + h[active, None, None] * d2g[active]
        return cost_hessian(spec) + 2 * r * terms.sum(axis=0)

    return Subproblem("penalty", r, fun, grad, hess)
