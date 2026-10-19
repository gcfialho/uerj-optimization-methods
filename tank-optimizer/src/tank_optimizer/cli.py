"""Command-line entry point: ``tank-optimizer serve`` (default) or ``tank-optimizer compare``."""

from __future__ import annotations

import argparse
import logging

from tank_optimizer.objective import SPEC, cost, volume
from tank_optimizer.service import run_comparison


def _compare(args: argparse.Namespace) -> None:
    payload = run_comparison(
        {"D0": args.d0, "L0": args.l0, "tol": args.tol, "max_iter": args.max_iter}
    )
    print(f"start D0={args.d0} L0={args.l0}  (cost {float(cost(args.d0, args.l0)):,.2f} $)")
    header = f"{'method':<8}{'D* [m]':>9}{'L* [m]':>9}{'V [m3]':>9}{'cost [$]':>11}"
    print(header + f"{'iter':>6}{'f evals':>9}  status")
    for name, r in payload["results"].items():
        print(
            f"{name:<8}{r['final_D']:>9.4f}{r['final_L']:>9.4f}"
            f"{float(volume(r['final_D'], r['final_L'])):>9.4f}{r['final_cost']:>11,.2f}"
            f"{r['iterations']:>6}{r['num_eval']:>9}  {r['message']}"
        )
    print(f"feasible volume range: [{SPEC.min_volume:.2f}, {SPEC.max_volume:.2f}] m3")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tank-optimizer", description="Cylindrical tank cost optimization."
    )
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="start the web UI (default)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5000)
    serve.add_argument("--debug", action="store_true", help="Flask debug mode (local use only)")
    compare = sub.add_parser("compare", help="run SD, Newton and DFP and print a table")
    compare.add_argument("--d0", type=float, default=0.5, help="initial diameter [m]")
    compare.add_argument("--l0", type=float, default=0.5, help="initial length [m]")
    compare.add_argument("--tol", type=float, default=1e-6)
    compare.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "compare":
        _compare(args)
        return

    from tank_optimizer.app import create_app

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 5000)
    create_app().run(host=host, port=port, debug=getattr(args, "debug", False))


if __name__ == "__main__":
    main()
