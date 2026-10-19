"""Nelder-Mead downhill simplex method with benchmark problems and animations."""

from nelder_mead.functions import FUNCTIONS, PROBLEMS, Benchmark, Problem, get_problem
from nelder_mead.optimizer import (
    Coefficients,
    Result,
    Step,
    default_simplex,
    minimize,
    simplex_volume,
)

__all__ = [
    "FUNCTIONS",
    "PROBLEMS",
    "Benchmark",
    "Coefficients",
    "Problem",
    "Result",
    "Step",
    "default_simplex",
    "get_problem",
    "minimize",
    "simplex_volume",
]
