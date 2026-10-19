"""Command-line interface: run Nelder-Mead on the bundled benchmarks.

Examples::

    nelder-mead --function rosenbrock --dim 2 --animate
    nelder-mead --all --animate --output-dir nelder-mead/results --gif-dir docs/assets
    nelder-mead --function sphere --dim 5 --compare-scipy
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from nelder_mead.functions import FUNCTIONS, PROBLEMS, Problem, get_problem
from nelder_mead.optimizer import Result, minimize

logger = logging.getLogger("nelder_mead")


def write_history(problem: Problem, result: Result, path: Path) -> None:
    """Write one CSV row per iteration: step, evaluations, best vertex and simplex size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = [f"x{i + 1}" for i in range(problem.dim)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteration", "step", "nfev", *coords, "f_best", "f_spread", "x_spread"])
        for k, (sim, fsim) in enumerate(zip(result.simplices, result.fvalues, strict=True)):
            writer.writerow(
                [
                    k,
                    result.steps[k].value,
                    result.nfev_history[k],
                    *(f"{v:.10g}" for v in sim[0]),
                    f"{fsim[0]:.6e}",
                    f"{fsim[-1] - fsim[0]:.3e}",
                    f"{abs(sim[1:] - sim[0]).max():.3e}",
                ]
            )
    logger.info("Saved history to %s", path)


def run(problem: Problem, args: argparse.Namespace) -> Result:
    result = minimize(
        problem.function.func,
        initial_simplex=problem.initial_simplex,
        xatol=args.xatol,
        fatol=args.fatol,
        max_iter=args.max_iter,
    )
    x = ", ".join(f"{v:.6f}" for v in result.x)
    print(
        f"{problem.slug:<18} nit={result.nit:<4} nfev={result.nfev:<4} f={result.fun:.3e}  "
        f"|x - x*|={problem.function.error(result.x):.2e}  x=({x})"
    )
    if not result.converged:
        logger.warning("%s: %s", problem.slug, result.message)

    if args.compare_scipy:
        _compare_scipy(problem, result, args)
    if args.save_history:
        write_history(problem, result, args.output_dir / f"{problem.slug}.csv")
    if args.animate or args.show:
        from nelder_mead.visualize import animate

        animate(problem, result, None if args.show else args.gif_dir / f"{problem.slug}.gif")
    return result


def _compare_scipy(problem: Problem, result: Result, args: argparse.Namespace) -> None:
    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError:
        logger.error("--compare-scipy needs SciPy (`uv sync` installs it with the dev group)")
        return
    ref = scipy_minimize(
        problem.function.func,
        problem.initial_simplex[0],
        method="Nelder-Mead",
        options={
            "initial_simplex": problem.initial_simplex,
            "xatol": args.xatol,
            "fatol": args.fatol,
            "maxiter": args.max_iter,
        },
    )
    print(
        f"{'  scipy':<18} nit={ref.nit:<4} nfev={ref.nfev:<4} f={ref.fun:.3e}  "
        f"|x - x_ours|={abs(ref.x - result.x).max():.2e}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nelder-mead",
        description="Run the Nelder-Mead simplex method on benchmark functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--function", "-f", choices=sorted(FUNCTIONS), default="rosenbrock")
    target.add_argument("--all", action="store_true", help="run every bundled demo problem")
    parser.add_argument("--dim", "-d", type=int, help="problem dimension (default: the demo's)")
    parser.add_argument("--xatol", type=float, default=1e-8, help="absolute tolerance on x")
    parser.add_argument("--fatol", type=float, default=1e-8, help="absolute tolerance on f")
    parser.add_argument("--max-iter", type=int, help="iteration budget (default 200 * dim)")
    parser.add_argument("--animate", action="store_true", help="save a GIF of the simplex")
    parser.add_argument("--show", action="store_true", help="open an interactive window instead")
    parser.add_argument(
        "--no-history", dest="save_history", action="store_false", help="skip the CSV history"
    )
    parser.add_argument("--compare-scipy", action="store_true", help="also run SciPy's solver")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="CSV directory")
    parser.add_argument("--gif-dir", type=Path, help="GIF directory (default: --output-dir)")
    parser.add_argument("--verbose", "-v", action="store_true", help="log every iteration")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.gif_dir = args.gif_dir or args.output_dir
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    try:
        problems = list(PROBLEMS.values()) if args.all else [get_problem(args.function, args.dim)]
    except ValueError as exc:
        parser.error(str(exc))
    for problem in problems:
        run(problem, args)


if __name__ == "__main__":
    main()
