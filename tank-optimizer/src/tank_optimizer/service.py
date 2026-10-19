"""Controller layer: gathers UI parameters, calls the optimization, and returns data."""
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Type, List

from models.optimizers import SteepestDescent, Newton, DFP, FunctionWrapper, BaseOptimizer
from models.objective import penalized_cost, cost

METHOD_MAP: Dict[str, Type[BaseOptimizer]] = {
    "SD": SteepestDescent,
    "Newton": Newton,
    "DFP": DFP,
}

def run_optimization(params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the optimization based on the interface parameters."""
    method: str = params.get("method", "SD")
    x0: NDArray[np.float64] = np.array([float(params["D0"]), float(params["L0"])])
    tol: float = float(params.get("tol", 1e-6))
    max_it: int = int(params.get("max_iter", 200))

    # Wrap the cost function to count evaluations
    fun: FunctionWrapper = FunctionWrapper(penalized_cost)

    OptimCls: Type[BaseOptimizer] = METHOD_MAP[method]
    
    # The `lr` step is no longer necessary, as all methods use line search
    opt: BaseOptimizer = OptimCls(fun, x0, tol=tol, max_iter=max_it)
        
    result = opt.optimize()

    # --- Prepare data for the View ---
    
    # 1. Optimization Results
    final_x: NDArray[np.float64] = result["x"]
    final_cost: float = cost(final_x[0], final_x[1])

    # 2. Contour Plot Data
    d_range: NDArray[np.float64] = np.linspace(0.1, 1.2, 50, dtype=np.float64)
    l_range: NDArray[np.float64] = np.linspace(0.1, 2.2, 50, dtype=np.float64)
    d_grid, l_grid = np.meshgrid(d_range, l_range)
    cost_grid: NDArray[np.float64] = np.zeros_like(d_grid, dtype=np.float64)
    
    for i in range(d_grid.shape[0]):
        for j in range(d_grid.shape[1]):
            cost_grid[i, j] = cost(d_grid[i, j], l_grid[i, j])

    # 3. Build the final payload
    payload: Dict[str, Any] = {
        # Results
        "history": np.array(result["history"], dtype=float).tolist(),
        "errors":  result["errors"],
        "final_D": final_x[0],
        "final_L": final_x[1],
        "final_cost": final_cost,
        "iterations": len(result["errors"]) - 1,
        "num_eval": result.get("fun_evals", 0),
        
        # Contour Plot Data
        "contour": {
            "d": d_range.tolist(),
            "l": l_range.tolist(),
            "cost": cost_grid.tolist()
        }
    }
    return payload

def run_comparison(params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the optimization for all three methods in comparison mode."""
    x0: NDArray[np.float64] = np.array([float(params["D0"]), float(params["L0"])])
    tol: float = float(params.get("tol", 1e-6))
    max_it: int = int(params.get("max_iter", 200))

    # Object to store aggregated results
    comparison_results: Dict[str, Any] = {}

    # Run each method
    for method_name, OptimCls in METHOD_MAP.items():
        fun = FunctionWrapper(penalized_cost)
        opt = OptimCls(fun, x0, tol=tol, max_iter=max_it)
        result = opt.optimize()

        final_x = result["x"]
        final_cost_val = cost(final_x[0], final_x[1])

        comparison_results[method_name] = {
            "history": np.array(result["history"], dtype=float).tolist(),
            "errors":  result["errors"],
            "final_D": final_x[0],
            "final_L": final_x[1],
            "final_cost": final_cost_val,
            "iterations": len(result["errors"]) - 1,
            "num_eval": result.get("fun_evals", 0),
        }

    # Calculate contour data (only once)
    d_range = np.linspace(0.1, 1.2, 50, dtype=np.float64)
    l_range = np.linspace(0.1, 2.2, 50, dtype=np.float64)
    d_grid, l_grid = np.meshgrid(d_range, l_range)
    cost_grid = np.zeros_like(d_grid, dtype=np.float64)
    
    for i in range(d_grid.shape[0]):
        for j in range(d_grid.shape[1]):
            cost_grid[i, j] = cost(d_grid[i, j], l_grid[i, j])

    # Build the final payload
    payload: Dict[str, Any] = {
        "results": comparison_results,
        "contour": {
            "d": d_range.tolist(),
            "l": l_range.tolist(),
            "cost": cost_grid.tolist()
        }
    }
    return payload