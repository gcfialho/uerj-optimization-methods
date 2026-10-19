# tank-optimizer

Minimum-cost design of a closed cylindrical steel tank under volume and size constraints. The problem is solved as a sequence of penalty and barrier subproblems (`constrained.py`), each minimized by Steepest Descent, Newton or DFP (quasi-Newton) with exact derivatives. SD and Newton use Armijo backtracking; DFP uses a strong-Wolfe line search.

The Flask + Plotly UI shows:

- the cost contours and the feasible region;
- each method's trajectory;
- the gradient-norm convergence;
- a side-by-side comparison of the three methods.

## Run

From the repository root, after `uv sync`:

```bash
uv run tank-optimizer                              # web UI on http://127.0.0.1:5050
uv run tank-optimizer serve --port 8080 --debug    # other port, Flask debug mode (local only)
uv run tank-optimizer compare --d0 0.5 --l0 0.5    # SD vs Newton vs DFP in the terminal
```

API: `POST /optimize` with `{D0, L0, method, tol, max_iter}`, and `POST /compare` with `{D0, L0, tol, max_iter}`.

## Layout

All modules are in `src/tank_optimizer/`:

- `objective.py`: the cost model $C(D, L)$, the constraints $g \le 0$, and the penalty and barrier subproblems with their exact derivatives.
- `constrained.py`: the SUMT driver. It runs an exterior penalty until the design is feasible, then barrier weights $\mu = 10 \to 0.01$.
- `optimizers.py`: SD, Newton and DFP, the Armijo and strong-Wolfe line searches, and evaluation counting.
- `service.py`: input validation and the JSON payload. `app.py` is the Flask app factory; `cli.py` provides `serve` and `compare`.
- `templates/index.html` and `static/js/main.js`: the UI.

The problem formulation and results table are in the [root README](../README.md).

## Changes from the original coursework version

The original version minimized one surrogate: the cost plus a log barrier
($\mu = 10^{-3}$) inside the feasible set, and the cost plus a quadratic penalty plus a
constant $10^{12}$ outside it, all with finite-difference derivatives. This had three
problems:

* At $f \approx 10^{12}$, the spacing between adjacent floats ($\approx 10^{-4}$) is larger than
  the change a $10^{-8}$ step produces. Central differences therefore returned a gradient of
  exactly zero, and the methods "converged" at infeasible designs. From $(0.5, 0.5)$,
  Steepest Descent stopped at $D = 1.002 > D_{\max}$ with $V = 0.717$, and Newton at
  $V = 0.718$. From $(0.98, 0.95)$, SD reported 5,340.94 \$, below the true optimum,
  for a tank that is too small.
* The gradient of the penalty branch ignores the constant, so the surrogate has a spurious
  stationary point outside the feasible set ($V \approx 0.7176$). Once the derivatives are
  exact, Newton converges to it.
* DFP updated $B$ whenever $|s^\top y| > 10^{-10}$, even when $s^\top y < 0$, so $B$ could lose
  positive definiteness. Fed with the $10^{20}$-sized finite-difference gradients produced at
  the boundary, it diverged from 3 of 8 starts (the default plus the seven distinct UI
  presets) to $D \approx 5 \times 10^9$ with a NaN
  gradient norm.

The penalty-then-barrier sequence with exact derivatives fixes all three. The line search no
longer re-evaluates $f(x)$ and the gradient at every backtracking step. The UI's replay
animations no longer cancel each other, and the UI is now in English.

Coursework for the Optimization course at UERJ (Instituto Politécnico), taught by Prof. [Gustavo Libotte](https://github.com/gustavolibotte).
