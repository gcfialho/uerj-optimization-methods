# nelder-mead

A from-scratch, N-dimensional implementation of the Nelder-Mead downhill simplex method
(no derivatives), with benchmark functions, per-iteration CSV histories and animations of the
simplex: triangles over contour plots in 2-D and tetrahedra in 3-D.

<p>
  <img src="../docs/assets/rosenbrock-2d.gif" width="49%" alt="Nelder-Mead on Rosenbrock (2-D)">
  <img src="../docs/assets/sphere-3d.gif" width="49%" alt="Nelder-Mead on the sphere function (3-D)">
</p>

More animations: [Booth](../docs/assets/booth-2d.gif), [Himmelblau](../docs/assets/himmelblau-2d.gif),
[shifted sphere (3-D)](../docs/assets/shifted-sphere-3d.gif).

## Algorithm

Each iteration sorts the $n+1$ vertices so that $f(x_1) \le \dots \le f(x_{n+1})$, computes
the centroid $\bar{x}$ of the best $n$ and tries, in order:

| Operation | Point | Accepted when |
| --- | --- | --- |
| Reflection | $x_r = \bar{x} + \alpha(\bar{x} - x_{n+1})$ | $f_1 \le f_r < f_n$ |
| Expansion | $x_e = \bar{x} + \gamma(x_r - \bar{x})$ | $f_r < f_1$ and $f_e < f_r$ (else keep $x_r$) |
| Outside contraction | $x_{oc} = \bar{x} + \rho(x_r - \bar{x})$ | $f_n \le f_r < f_{n+1}$ and $f_{oc} \le f_r$ |
| Inside contraction | $x_{ic} = \bar{x} + \rho(x_{n+1} - \bar{x})$ | $f_r \ge f_{n+1}$ and $f_{ic} < f_{n+1}$ |
| Shrink | $x_i \leftarrow x_1 + \sigma(x_i - x_1)$ | a contraction failed |

Coefficients: $\alpha = 1$, $\gamma = 2$, $\rho = 0.5$, $\sigma = 0.5$ (configurable through
`Coefficients`). The run stops when both
$\max_i \lVert x_i - x_1\rVert_\infty \le$ `xatol` and $\max_i |f_i - f_1| \le$ `fatol`
(default $10^{-8}$), or when the budget of $200n$ iterations / evaluations is spent. This is the
formulation of Lagarias et al. (1998), the same one SciPy uses, so with the same initial simplex
and tolerances both produce the same iterates (checked in the tests).

## Usage

```bash
uv run nelder-mead --function rosenbrock --dim 2 --animate   # writes output/rosenbrock-2d.{csv,gif}
uv run nelder-mead --function sphere --dim 6 --compare-scipy # any n for sphere / rosenbrock
uv run nelder-mead --function himmelblau --show              # interactive window
uv run nelder-mead --all --animate --output-dir nelder-mead/results --gif-dir docs/assets
uv run nelder-mead --help
```

```python
from nelder_mead import minimize, get_problem

problem = get_problem("rosenbrock", 2)
result = minimize(problem.function.func, initial_simplex=problem.initial_simplex)
result.x, result.fun, result.nit, result.nfev  # plus result.simplices / result.steps
```

Bundled functions (all vectorized over the last axis): `sphere`, `shifted-sphere` and
`rosenbrock` (any $n \ge 2$), `booth` and `himmelblau` (2-D).

## Results

All five bundled demos, default tolerances (`xatol = fatol = 1e-8`), compared with
`scipy.optimize.minimize(method="Nelder-Mead")` from the same initial simplex
(`uv run nelder-mead --all --compare-scipy`):

| Problem | Iterations | f evals | Final f | Distance to x* | SciPy nit / nfev | Max. difference from SciPy x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rosenbrock 2-D | 126 | 232 | 9.7e-18 | 1.7e-9 | 127 / 232 | 1.4e-14 |
| Booth 2-D | 66 | 130 | 5.9e-17 | 6.1e-9 | 67 / 130 | 8.9e-16 |
| Himmelblau 2-D | 66 | 129 | 1.9e-16 | 3.8e-9 (to (3, 2)) | 67 / 129 | 4.4e-16 |
| Sphere 3-D | 126 | 235 | 2.2e-17 | 4.7e-9 | 127 / 235 | 1.3e-14 |
| Shifted sphere 3-D | 122 | 228 | 2.9e-17 | 5.4e-9 | 123 / 228 | 4.0e-15 |

SciPy starts its iteration counter at 1, so its `nit` is one higher for the same number
of simplex updates. The function-evaluation counts are identical.

The per-iteration histories are in [`results/`](results/) (one CSV row per iteration: step
taken, evaluations so far, best vertex, f-spread and simplex size). For reference, the
original coursework version needed 399 evaluations on Rosenbrock and stopped at
f = 8.3e-7, 2.0e-3 away from the minimum. That version re-evaluated every vertex on every
iteration, had no outside contraction, and stopped on the standard deviation of f alone.

## Layout

```
src/nelder_mead/
├── optimizer.py   # minimize(), Coefficients, Result, Step, default_simplex, simplex_volume
├── functions.py   # benchmark functions, known minimizers, bundled demo problems
├── visualize.py   # 2-D/3-D animations (GIF via Pillow, MP4 via ffmpeg)
└── cli.py         # `nelder-mead` command
results/           # CSV histories of the bundled demos (one row per iteration)
tests/             # convergence, SciPy equivalence, simplex invariants, CLI
```
