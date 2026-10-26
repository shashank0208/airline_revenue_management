"""
EMSR-b (Expected Marginal Seat Revenue — version b) nested protection levels.

Theory
------
For fare classes indexed 1 (highest) … n (lowest), EMSR-b protects seats
for the higher classes against the next-lower class.

Protection level y_j for the *aggregate* demand of classes 1..j against
class j+1 is the solution to:

    P(D_{1..j} > y_j)  =  f_{j+1} / fbar_j

where fbar_j is the *demand-weighted* average fare of classes 1..j, and
D_{1..j} ~ N(μ_{1..j}, σ²_{1..j}) is the aggregate demand (sum of
independent normals for each class in 1..j).

Booking limit for class j+1 is then  BL_{j+1} = C - y_j, where C is
aircraft capacity.  The highest class has BL_1 = C (always open).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def emsr_b_protection_levels(
    fares: list[float],
    means: list[float],
    std_devs: list[float],
) -> list[int]:
    """
    Compute nested protection levels using EMSR-b.

    Parameters
    ----------
    fares : list[float]
        Fares for classes, **highest fare first** (Y, B, M, Q).
    means : list[float]
        Mean demand per class, same ordering.
    std_devs : list[float]
        Std-dev of demand per class, same ordering.

    Returns
    -------
    protection_levels : list[int]
        y_j for j = 1 … n-1 (length n-1).  y_j is the number of seats
        to protect for classes 1..j against class j+1.
    """
    n = len(fares)
    if n < 2:
        return []

    protection: list[int] = []

    for j in range(1, n):
        # Aggregate demand of classes 1..j
        mu_agg = sum(means[:j])
        sigma_agg = np.sqrt(sum(s ** 2 for s in std_devs[:j]))

        # Demand-weighted average fare of classes 1..j
        total_demand = sum(means[:j])
        if total_demand == 0:
            fbar = fares[0]
        else:
            fbar = sum(f * d for f, d in zip(fares[:j], means[:j])) / total_demand

        # Fare ratio
        if fbar == 0:
            protection.append(0)
            continue
        ratio = fares[j] / fbar

        if ratio >= 1.0:
            # Lower class fare ≥ weighted average of higher classes → protect 0
            protection.append(0)
            continue

        # Solve P(D_agg > y_j) = ratio  ⟹  y_j = μ + σ·Φ⁻¹(1 - ratio)
        if sigma_agg < 1e-9:
            # Deterministic demand: protect exactly the mean
            protection.append(max(0, int(round(mu_agg))))
            continue

        y = mu_agg + sigma_agg * norm.ppf(1.0 - ratio)
        protection.append(max(0, int(round(y))))

    return protection


def protection_to_booking_limits(
    protection_levels: list[int],
    capacity: int,
) -> list[int]:
    """
    Convert nested protection levels to booking limits per class.

    Parameters
    ----------
    protection_levels : list[int]
        y_1, y_2, …, y_{n-1}  (length n-1).
    capacity : int
        Aircraft capacity.

    Returns
    -------
    booking_limits : list[int]
        BL for each class (length n = len(protection_levels)+1).
        BL[0] = capacity  (highest fare class is always open).
        BL[j] = capacity - y_{j-1}  for j >= 1.
    """
    n = len(protection_levels) + 1
    bl: list[int] = [capacity]  # class 0 (highest fare)
    for y in protection_levels:
        bl.append(max(0, capacity - y))
    return bl


def compute_emsr_b(
    fares: list[float],
    means: list[float],
    std_devs: list[float],
    capacity: int,
) -> dict:
    """
    Convenience wrapper: compute protection levels and booking limits.

    Parameters are **highest fare first**.

    Returns dict with keys:
        protection_levels, booking_limits, fares, means, std_devs
    """
    pl = emsr_b_protection_levels(fares, means, std_devs)
    bl = protection_to_booking_limits(pl, capacity)
    return {
        "protection_levels": pl,
        "booking_limits": bl,
        "fares": fares,
        "means": means,
        "std_devs": std_devs,
        "capacity": capacity,
    }
