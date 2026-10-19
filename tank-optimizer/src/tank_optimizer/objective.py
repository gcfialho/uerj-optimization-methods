"""Cost functions and constraints for the tank optimization problem."""
import numpy as np
from numpy.typing import NDArray

# Parameters (Table 1)
V0: float = 0.8            # m³ (required volume)
t: float = 0.03            # m  (thickness – 3 cm)
rho: float = 8000          # kg/m³ (density)
L_MAX: float = 2.0         # m (max length)
D_MAX: float = 1.0         # m (max diameter)
c_material: float = 4.5    # $/kg (material cost)
c_weld: float = 20.0       # $/m (welding cost)


def cost(D: float, L: float) -> float:
    """Total cost of the tank for a given diameter D and length L."""
    # Mass calculation
    V_cyl: float = L * np.pi * ((D/2 + t)**2 - (D/2)**2)
    V_plates: float = 2 * np.pi * ((D/2 + t)**2) * t
    mass: float = rho * (V_cyl + V_plates)

    # Weld length
    weld_length: float = 4 * np.pi * (D + t)

    return c_material * mass + c_weld * weld_length


def constraints(D: float, L: float) -> NDArray[np.float64]:
    """Returns the vector g(x) <= 0 of the constraints."""
    vol: float = (np.pi * D**2 * L) / 4
    g1: float  = (0.9 * V0) - vol          # min volume
    g2: float  = vol - (1.1 * V0)          # max volume
    g3: float  = L - L_MAX                 # max length
    g4: float  = D - D_MAX                 # max diameter
    return np.array([g1, g2, g3, g4])

def penalized_cost(x: NDArray[np.float64], r_penalty: float = 1e6, t_barrier: float = 1e-3) -> float:
    """Cost function with an exterior penalty and an interior barrier."""
    D, L = x[0], x[1]

    # Unbreakable barrier for non-positive dimensions.
    # This is the most critical check for stability.
    if D <= 0 or L <= 0:
        return np.inf
    
    # Original cost
    base_cost: float = cost(D, L)

    # Penalties for constraint violations g(x) <= 0 (Barrier Method)
    g = constraints(D, L)
    
    # Check if any constraint is "too" violated or at the boundary.
    # If g(x) >= 0, the log is not defined. Return a very high value.
    if np.any(g >= 0):
        # Use a quadratic penalty to push back into the feasible region
        return base_cost + r_penalty * np.sum(np.maximum(0, g)**2) + 1e12

    # If g(x) < 0 for all constraints, apply the logarithmic barrier
    barrier_cost = -t_barrier * np.sum(np.log(-g))

    # The penalty for non-positive D and L is now handled by the np.inf barrier,
    # but we keep a small penalty in case the gradient is calculated
    # very close to zero, although the main barrier should prevent this.
    penalty_non_positive = 0.0
    if D <= 1e-6:
        penalty_non_positive += r_penalty * (abs(D) + 0.1)**2
    if L <= 1e-6:
        penalty_non_positive += r_penalty * (abs(L) + 0.1)**2
        
    return base_cost + barrier_cost + penalty_non_positive