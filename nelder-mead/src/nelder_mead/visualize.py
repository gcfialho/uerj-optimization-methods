"""Animations of the simplex evolution.

Left panel: the simplex (triangle over log-scaled contours in 2-D, tetrahedron in 3-D) with
a fading trail of recent simplices and the path of the best vertex. Right panel: best value
and simplex size per iteration on a log scale.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nelder_mead.functions import Problem
from nelder_mead.optimizer import FloatArray, Result

__all__ = ["animate"]

logger = logging.getLogger(__name__)

TETRA_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
TRAIL = 12  # number of previous simplices drawn behind the current one
SIMPLEX_COLOR = "#e4572e"
PATH_COLOR = "#ffc914"

Update = Callable[[int], None]


def _frame_indices(n_states: int, max_frames: int, hold: int) -> list[int]:
    """Evenly subsample long runs and hold the final simplex for a few frames."""
    if n_states > max_frames:
        idx = np.unique(np.linspace(0, n_states - 1, max_frames).round().astype(int)).tolist()
    else:
        idx = list(range(n_states))
    return idx + [n_states - 1] * hold


def _auto_bounds(problem: Problem, result: Result, pad: float = 0.15) -> list[tuple[float, float]]:
    points = np.vstack([*result.simplices, problem.function.minimizers(problem.dim)])
    lo, hi = points.min(axis=0), points.max(axis=0)
    margin = np.maximum((hi - lo) * pad, 0.5)
    return list(zip(lo - margin, hi + margin, strict=True))


def _trail_alphas(k: int) -> list[float]:
    start = max(0, k - TRAIL)
    return [0.6 * (i - start + 1) / (k - start + 1) for i in range(start, k)]


def _convergence_panel(ax: Axes, result: Result) -> Update:
    """Static curves for the whole run plus a marker that follows the current iteration."""
    iters = np.arange(len(result.simplices))
    floor = 1e-20
    f_best = np.maximum([fv[0] for fv in result.fvalues], floor)
    size = np.maximum([np.abs(s[1:] - s[0]).max() for s in result.simplices], floor)
    ax.semilogy(iters, f_best, color="#2e86ab", lw=1.5, label="$f(x_1)$")
    ax.semilogy(iters, size, color="#a23b72", lw=1.5, label=r"$\max_i \|x_i - x_1\|_\infty$")
    ax.set(xlabel="iteration", xlim=(0, iters[-1]), title="convergence")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    cursor = ax.axvline(0, color="0.3", lw=1, ls="--")
    (dot_f,) = ax.plot([], [], "o", color="#2e86ab", ms=5)
    (dot_s,) = ax.plot([], [], "o", color="#a23b72", ms=5)

    def update(k: int) -> None:
        cursor.set_xdata([k, k])
        dot_f.set_data([k], [f_best[k]])
        dot_s.set_data([k], [size[k]])

    return update


def _simplex_panel_2d(fig: Figure, spec, problem: Problem, result: Result) -> tuple[Axes, Update]:
    ax = fig.add_subplot(spec)
    (x_min, x_max), (y_min, y_max) = problem.bounds or _auto_bounds(problem, result)
    xs, ys = np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    z = np.log1p(problem.function.func(np.stack(np.meshgrid(xs, ys), axis=-1)))
    ax.contourf(xs, ys, z, levels=30, cmap="viridis")
    ax.contour(xs, ys, z, levels=15, colors="white", linewidths=0.3, alpha=0.4)
    minima = problem.function.minimizers(2)
    ax.plot(minima[:, 0], minima[:, 1], "x", color="white", ms=8, mew=2, label="known minimum")
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), xlabel="$x_1$", ylabel="$x_2$")
    ax.set_aspect("equal", adjustable="box")

    trail = PolyCollection([], facecolors="none", linewidths=0.8)
    ax.add_collection(trail)
    current = Polygon(result.simplices[0], closed=True, fc="none", ec=SIMPLEX_COLOR, lw=2.2)
    ax.add_patch(current)
    best = np.array([s[0] for s in result.simplices])
    (path,) = ax.plot([], [], "-", color=PATH_COLOR, lw=1.2, label="best vertex")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    def update(k: int) -> None:
        current.set_xy(result.simplices[k])
        trail.set_verts(result.simplices[max(0, k - TRAIL) : k])
        trail.set_edgecolor([(1, 1, 1, a) for a in _trail_alphas(k)])
        path.set_data(best[: k + 1, 0], best[: k + 1, 1])

    return ax, update


def _simplex_panel_3d(fig: Figure, spec, problem: Problem, result: Result) -> tuple[Axes, Update]:
    ax: Axes3D = fig.add_subplot(spec, projection="3d")
    x_lim, y_lim, z_lim = problem.bounds or _auto_bounds(problem, result)
    ax.set(xlim=x_lim, ylim=y_lim, zlim=z_lim, xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")
    ax.view_init(elev=22, azim=-55)

    minimum = problem.function.minimizers(3)[0]
    ax.scatter(*minimum, marker="x", color="black", s=60, label="known minimum")
    best = np.array([s[0] for s in result.simplices])
    (path,) = ax.plot([], [], [], "-", color="#1f77b4", lw=1.4, label="best vertex")

    def faces(tetra: FloatArray) -> list[FloatArray]:
        return [tetra[list(face)] for face in TETRA_FACES]

    tetra = Poly3DCollection(
        faces(result.simplices[0]), facecolor=(0.9, 0.34, 0.18, 0.35), edgecolor=SIMPLEX_COLOR
    )
    ax.add_collection3d(tetra)
    ax.legend(loc="upper left", fontsize=8)

    def update(k: int) -> None:
        tetra.set_verts(faces(result.simplices[k]))
        path.set_data_3d(best[: k + 1, 0], best[: k + 1, 1], best[: k + 1, 2])

    return ax, update


def animate(
    problem: Problem,
    result: Result,
    path: Path | None = None,
    *,
    fps: int = 8,
    max_frames: int = 90,
    hold: int = 12,
    dpi: int = 72,
) -> None:
    """Render the run as an animation; save it to ``path`` (``.gif`` or ``.mp4``) or show it."""
    if problem.dim not in (2, 3):
        msg = f"animations are only available for 2-D and 3-D problems, not {problem.dim}-D"
        raise ValueError(msg)
    frames = _frame_indices(len(result.simplices), max_frames, hold)
    fig = plt.figure(figsize=(10, 4.6), dpi=dpi, layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.25, 1))
    panel = _simplex_panel_2d if problem.dim == 2 else _simplex_panel_3d
    ax, update_simplex = panel(fig, grid[0], problem, result)
    update_convergence = _convergence_panel(fig.add_subplot(grid[1]), result)
    title = ax.set_title("", fontsize=10)

    def update(k: int) -> None:
        update_simplex(k)
        update_convergence(k)
        title.set_text(
            f"{problem.function.name} ({problem.dim}-D)  iter {k}/{result.nit}  "
            f"{result.steps[k].value.replace('_', ' ')}"
        )

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    if path is None:
        plt.show()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: animation.AbstractMovieWriter
    if path.suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)
    logger.info("Saved animation to %s (%.0f kB)", path, path.stat().st_size / 1024)
