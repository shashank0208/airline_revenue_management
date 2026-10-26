"""
Linear programming seat allocation using PuLP.

Maximise total revenue  ∑ fare_i · x_i
subject to:
    ∑ x_i  <=  capacity           (total seats)
    0  <=  x_i  <=  demand_i      (per-class demand cap)
    x_i integer

The LP can be re-solved during the booking horizon as observed demand
updates the forecast.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pulp
    PULP_AVAILABLE = True
    _PULP_MAJOR = int(pulp.__version__.split(".")[0])
except ImportError:
    PULP_AVAILABLE = False
    _PULP_MAJOR = 0


def _create_variable(prob, name: str, low: float, up: float, cat: str = "Integer"):
    """Create an LpVariable compatible with PuLP 2.x and 4.x."""
    if _PULP_MAJOR >= 4:
        return pulp.LpVariable.fromDict(prob, {
            "name": name,
            "lowBound": low,
            "upBound": up,
            "cat": cat,
        })
    else:
        return pulp.LpVariable(name, lowBound=low, upBound=up, cat=cat)


def _get_solver():
    """Get the best available solver."""
    if _PULP_MAJOR >= 4:
        for solver_name in ["HiGHS", "HiGHS_CMD", "COIN_CMD", "GLPK_CMD"]:
            try:
                solver_cls = getattr(pulp.apis, solver_name, None)
                if solver_cls is not None:
                    solver = solver_cls(msg=0)
                    if solver.available():
                        return solver
            except Exception:
                continue
        return None
    else:
        return pulp.PULP_CBC_CMD(msg=0)


def _get_status_string(prob, solve_result) -> str:
    """Extract status string from solve result, handling PuLP 2.x and 4.x."""
    if _PULP_MAJOR >= 4:
        # In 4.x, solve() returns a result object with a string repr
        result_str = str(solve_result)
        if "Optimal" in result_str:
            return "Optimal"
        elif "Infeasible" in result_str:
            return "Infeasible"
        else:
            return result_str.split("\n")[1] if "\n" in result_str else str(solve_result)
    else:
        return pulp.LpStatus[prob.status]


def solve_lp(
    fares: list[float],
    demand_forecasts: list[float],
    capacity: int,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Solve the seat-allocation LP.

    Parameters
    ----------
    fares : list[float]
        Fare for each class (any ordering, but consistent with demand_forecasts).
    demand_forecasts : list[float]
        Forecast demand for each class.
    capacity : int
        Aircraft capacity.
    class_names : list[str], optional
        Labels for classes (for readability).

    Returns
    -------
    dict with keys: allocations, revenue, status, class_names
    """
    if not PULP_AVAILABLE:
        return _greedy_allocation(fares, demand_forecasts, capacity, class_names)

    n = len(fares)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n)]

    prob = pulp.LpProblem("SeatAllocation", pulp.LpMaximize)

    # Decision variables
    x = []
    for i in range(n):
        ub = max(0, int(np.ceil(demand_forecasts[i])))
        var = _create_variable(prob, f"x_{class_names[i]}", 0, ub, "Integer")
        x.append(var)

    # Objective
    prob += pulp.lpSum(fares[i] * x[i] for i in range(n)), "TotalRevenue"

    # Capacity constraint
    prob += pulp.lpSum(x) <= capacity, "Capacity"

    # Solve
    solver = _get_solver()
    try:
        if solver is not None:
            solve_result = prob.solve(solver)
        else:
            solve_result = prob.solve()
    except Exception:
        return _greedy_allocation(fares, demand_forecasts, capacity, class_names)

    allocations = [int(xi.varValue or 0) for xi in x]

    # Get objective value
    try:
        revenue = float(pulp.value(prob.objective) or 0)
    except Exception:
        revenue = sum(fares[i] * allocations[i] for i in range(n))

    status = _get_status_string(prob, solve_result)

    return {
        "allocations": allocations,
        "revenue": revenue,
        "status": status,
        "class_names": class_names,
    }


def _greedy_allocation(
    fares: list[float],
    demand_forecasts: list[float],
    capacity: int,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Greedy fallback when PuLP is unavailable: fill highest fare first."""
    n = len(fares)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n)]

    order = sorted(range(n), key=lambda i: fares[i], reverse=True)
    allocations = [0] * n
    remaining = capacity

    for i in order:
        alloc = min(int(np.ceil(demand_forecasts[i])), remaining)
        allocations[i] = alloc
        remaining -= alloc

    revenue = sum(fares[i] * allocations[i] for i in range(n))

    return {
        "allocations": allocations,
        "revenue": revenue,
        "status": "Greedy-Fallback",
        "class_names": class_names,
    }


def lp_booking_limits(
    fares: list[float],
    demand_forecasts: list[float],
    capacity: int,
    class_names: Optional[list[str]] = None,
) -> list[int]:
    """
    Derive nested booking limits from the LP solution.

    For classes ordered highest fare first (Y, B, M, Q):
      BL[0] = capacity              (highest class always open)
      BL[i] = capacity - sum(allocs[0..i-1])  (protect for higher classes)

    Parameters are ordered **highest fare first** (same as EMSR-b).
    """
    result = solve_lp(fares, demand_forecasts, capacity, class_names)
    allocs = result["allocations"]
    n = len(allocs)

    bl = [capacity]  # class 0 (highest fare) always open
    cumulative_higher = 0
    for i in range(1, n):
        cumulative_higher += allocs[i - 1]
        bl.append(max(0, capacity - cumulative_higher))
    return bl

