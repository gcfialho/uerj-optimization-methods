"""Implementations of optimization algorithms: Steepest Descent, Newton, and DFP."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Type

import numpy as np
from numpy.typing import NDArray

# --- Wrapper for Evaluation Counting ---
class FunctionWrapper:
    """Wraps the objective function to count the number of evaluations."""
    def __init__(self, fun: Callable[[NDArray[np.float64]], float]):
        self.fun = fun
        self.eval_count = 0
        self.grad_eval_count = 0

    def __call__(self, x: NDArray[np.float64]) -> float:
        self.eval_count += 1
        return self.fun(x)

    def grad(self, x: NDArray[np.float64], delta: float = 1e-8) -> NDArray[np.float64]:
        self.grad_eval_count += 1
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            x_plus_delta = np.copy(x)
            x_plus_delta[i] += delta
            x_minus_delta = np.copy(x)
            x_minus_delta[i] -= delta
            grad[i] = (self(x_plus_delta) - self(x_minus_delta)) / (2 * delta)
        return grad

# --- Base Optimizer Class ---
class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""
    def __init__(self, fun: FunctionWrapper, x0: NDArray[np.float64], tol: float, max_iter: int):
        self.fun = fun
        self.x = np.copy(x0)
        self.tol = tol
        self.max_iter = max_iter
        self.history: List[NDArray[np.float64]] = [np.copy(x0)]
        self.errors: List[float] = []

    def _line_search(self, x: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """Performs backtracking line search to find a suitable step size."""
        alpha = 1.0
        c = 1e-4
        rho = 0.5
        g = self.fun.grad(x)
        
        # The check for the descent direction has been moved to the `optimize` loop.
        while self.fun(x + alpha * p) > self.fun(x) + c * alpha * (p @ g):
            alpha *= rho
            if alpha < 1e-10:
                return 1e-10
        return alpha

    def optimize(self) -> Dict[str, Any]:
        """Main optimization loop."""
        k = 0
        g = self.fun.grad(self.x)
        self.errors.append(float(np.linalg.norm(g)))

        while np.linalg.norm(g) > self.tol and k < self.max_iter:
            p = self._get_direction(g)
            
            # Ensures that p is a descent direction.
            # If not, use the gradient direction as a fallback for the iteration.
            if np.dot(p, g) >= 0:
                p = -g

            alpha = self._line_search(self.x, p)
            self.x += alpha * p
            self.history.append(np.copy(self.x))
            g = self.fun.grad(self.x)
            self.errors.append(float(np.linalg.norm(g)))
            k += 1

        return {
            "x": self.x, "fun": self.fun(self.x), "grad": g,
            "history": self.history, "errors": self.errors,
            "fun_evals": self.fun.eval_count
        }

    @abstractmethod
    def _get_direction(self, g: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculates the search direction (specific to each method)."""
        pass

# --- Method Implementations ---
class SteepestDescent(BaseOptimizer):
    """Steepest Descent with line search."""
    def _get_direction(self, g: NDArray[np.float64]) -> NDArray[np.float64]:
        return -g

class Newton(BaseOptimizer):
    """Newton's method with line search and fallback."""
    def _get_hessian(self, x: NDArray[np.float64], delta: float = 1e-5) -> NDArray[np.float64]:
        n = len(x)
        H = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                x_pp = x.copy(); x_pp[i] += delta; x_pp[j] += delta
                x_pm = x.copy(); x_pm[i] += delta; x_pm[j] -= delta
                x_mp = x.copy(); x_mp[i] -= delta; x_mp[j] += delta
                x_mm = x.copy(); x_mm[i] -= delta; x_mm[j] -= delta
                H[i, j] = (self.fun(x_pp) - self.fun(x_pm) - self.fun(x_mp) + self.fun(x_mm)) / (4 * delta**2)
        return H

    def _get_direction(self, g: NDArray[np.float64]) -> NDArray[np.float64]:
        H = self._get_hessian(self.x)
        try:
            # Using 'solve' is numerically more stable and efficient than 'inv'
            p = np.linalg.solve(H, -g).astype(np.float64)
            # Safety check: if the step is too large, the Hessian
            # is likely ill-conditioned. Use the gradient as a fallback.
            if np.linalg.norm(p) > 100:
                return -g
            return p
        except np.linalg.LinAlgError:
            return -g # Fallback to gradient if the Hessian is singular

class DFP(BaseOptimizer):
    """Davidon-Fletcher-Powell (DFP) method."""
    def __init__(self, fun: FunctionWrapper, x0: NDArray[np.float64], tol: float, max_iter: int):
        super().__init__(fun, x0, tol, max_iter)
        self.B = np.eye(len(x0))  # Approximation of the Inverse Hessian

    def optimize(self) -> Dict[str, Any]:
        k = 0
        g = self.fun.grad(self.x)
        self.errors.append(float(np.linalg.norm(g)))

        while np.linalg.norm(g) > self.tol and k < self.max_iter:
            p = -self.B @ g
            
            # Safety check: ensure p is a descent direction.
            # If the B matrix is not positive definite, the direction might be wrong.
            if np.dot(p, g) >= 0:
                # Fallback to steepest descent and reset the Hessian approximation
                p = -g
                self.B = np.eye(len(self.x))
            
            x_old = self.x
            g_old = g
            
            alpha = self._line_search(self.x, p)
            self.x = self.x + alpha * p
            self.history.append(np.copy(self.x))
            
            g = self.fun.grad(self.x)
            self.errors.append(float(np.linalg.norm(g)))

            s = self.x - x_old
            y = g - g_old

            s_dot_y = s @ y
            y_B_y = y @ self.B @ y

            # Update the Hessian approximation (B) only if it is numerically stable
            if np.abs(s_dot_y) > 1e-10 and np.abs(y_B_y) > 1e-10:
                term1 = np.outer(s, s) / s_dot_y
                term2 = - (self.B @ np.outer(y, y) @ self.B) / y_B_y
                self.B += term1 + term2
            
            k += 1

        return {
            "x": self.x, "fun": self.fun(self.x), "grad": g,
            "history": self.history, "errors": self.errors,
            "fun_evals": self.fun.eval_count
        }

    def _get_direction(self, g: NDArray[np.float64]) -> NDArray[np.float64]:
        # In DFP, the direction is calculated directly in the main loop.
        # This method is not used but needs to be implemented.
        return np.zeros_like(g)