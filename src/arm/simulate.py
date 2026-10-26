"""
Monte Carlo booking simulation with common random numbers (CRN).

Booking requests arrive according to a time-varying Poisson process over
a 90-day booking horizon.  Each policy (FCFS, EMSR-b, LP) sees the
identical request stream within a single run and accepts or rejects
requests based on its booking limits.

Time profile: low-fare classes (Q) book early; high-fare classes (Y)
book late.  The arrival rate for class i on day d is modulated by a
Beta CDF so that the *cumulative* fraction of demand by day d follows
Beta(alpha_i, beta_i).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist

from arm.emsr import compute_emsr_b
from arm.lp import lp_booking_limits
from arm.metrics import PolicyMetrics


def _arrival_rates(
    total_demand: list[float],
    booking_horizon: int,
    profile_params: dict[str, dict],
    fare_class_codes: list[str],
) -> np.ndarray:
    """
    Compute expected arrivals per (class, day).

    Returns shape (n_classes, booking_horizon).
    Day 0 = first day of booking window (90 days before departure).
    Day booking_horizon-1 = departure day.
    """
    n_classes = len(fare_class_codes)
    rates = np.zeros((n_classes, booking_horizon))

    for i, code in enumerate(fare_class_codes):
        params = profile_params[code]
        a, b = params["alpha"], params["beta"]
        # CDF at each day boundary
        t = np.linspace(0, 1, booking_horizon + 1)
        cdf = beta_dist.cdf(t, a, b)
        # Daily increments
        daily_frac = np.diff(cdf)
        rates[i, :] = total_demand[i] * daily_frac

    return rates


def generate_request_stream(
    total_demand: list[float],
    booking_horizon: int,
    profile_params: dict[str, dict],
    fare_class_codes: list[str],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Generate a single booking-request stream using Poisson arrivals.

    Returns a list of (day, class_index) tuples sorted by day.
    """
    rates = _arrival_rates(total_demand, booking_horizon, profile_params, fare_class_codes)
    n_classes = len(fare_class_codes)
    requests: list[tuple[int, int]] = []

    for day in range(booking_horizon):
        for cls in range(n_classes):
            n_arrivals = rng.poisson(rates[cls, day])
            requests.extend([(day, cls)] * n_arrivals)

    # Sort by day (stable — within a day, order doesn't matter much)
    requests.sort(key=lambda x: x[0])
    return requests


def simulate_fcfs(
    requests: list[tuple[int, int]],
    capacity: int,
    n_classes: int,
    fares: list[float],
) -> dict:
    """First Come, First Served — accept every request until full."""
    seats_sold = [0] * n_classes
    total_sold = 0
    spilled = 0
    revenue = 0.0

    for _day, cls in requests:
        if total_sold < capacity:
            seats_sold[cls] += 1
            total_sold += 1
            revenue += fares[cls]
        else:
            spilled += 1

    return {"revenue": revenue, "seats_sold": sum(seats_sold), "spill": spilled}


def simulate_emsr(
    requests: list[tuple[int, int]],
    capacity: int,
    n_classes: int,
    fares: list[float],
    booking_limits: list[int],
) -> dict:
    """Accept/reject based on EMSR-b booking limits (nested)."""
    # booking_limits[i] = max seats that can be sold to classes i..n-1
    # Classes are ordered highest fare first (index 0 = Y, last = Q)
    seats_sold = [0] * n_classes
    total_sold = 0
    spilled = 0
    revenue = 0.0

    for _day, cls in requests:
        # Must respect both total capacity AND per-class booking limit
        sold_in_this_and_lower = sum(seats_sold[cls:])
        if total_sold < capacity and sold_in_this_and_lower < booking_limits[cls]:
            seats_sold[cls] += 1
            total_sold += 1
            revenue += fares[cls]
        else:
            spilled += 1

    return {"revenue": revenue, "seats_sold": total_sold, "spill": spilled}


def simulate_lp(
    requests: list[tuple[int, int]],
    capacity: int,
    n_classes: int,
    fares: list[float],
    booking_limits: list[int],
) -> dict:
    """Accept/reject based on LP-derived booking limits (nested)."""
    # Same logic as EMSR but with LP-derived limits
    return simulate_emsr(requests, capacity, n_classes, fares, booking_limits)


def run_simulation(
    config: dict,
    route_name: str,
    demand_means: list[float],
    demand_stds: list[float],
    fares: list[float],
    fare_class_codes: list[str],
    capacity: int,
    num_runs: int = 500,
    base_seed: int = 42,
) -> dict[str, PolicyMetrics]:
    """
    Run Monte Carlo simulation for a single route.

    Parameters are ordered **highest fare first** (Y, B, M, Q).

    Returns dict mapping policy name → PolicyMetrics.
    """
    booking_horizon = config.get("booking_horizon_days", 90)
    profile_params = config.get("booking_time_profile", {})

    # Compute booking limits
    emsr_result = compute_emsr_b(fares, demand_means, demand_stds, capacity)
    emsr_bl = emsr_result["booking_limits"]

    lp_bl = lp_booking_limits(fares, demand_means, capacity, class_names=fare_class_codes)

    n_classes = len(fares)

    # Storage
    fcfs_rev = np.zeros(num_runs)
    fcfs_sold = np.zeros(num_runs)
    fcfs_spill = np.zeros(num_runs)

    emsr_rev = np.zeros(num_runs)
    emsr_sold = np.zeros(num_runs)
    emsr_spill = np.zeros(num_runs)

    lp_rev = np.zeros(num_runs)
    lp_sold = np.zeros(num_runs)
    lp_spill = np.zeros(num_runs)

    for run in range(num_runs):
        # Common random numbers: same seed → same request stream
        rng = np.random.default_rng(base_seed + run)

        # Total demand for this run drawn from normal distribution
        total_demand = [max(0, rng.normal(m, s)) for m, s in zip(demand_means, demand_stds)]

        requests = generate_request_stream(
            total_demand, booking_horizon, profile_params, fare_class_codes, rng
        )

        # --- FCFS ---
        r = simulate_fcfs(requests, capacity, n_classes, fares)
        fcfs_rev[run] = r["revenue"]
        fcfs_sold[run] = r["seats_sold"]
        fcfs_spill[run] = r["spill"]

        # --- EMSR-b ---
        r = simulate_emsr(requests, capacity, n_classes, fares, emsr_bl)
        emsr_rev[run] = r["revenue"]
        emsr_sold[run] = r["seats_sold"]
        emsr_spill[run] = r["spill"]

        # --- LP ---
        r = simulate_lp(requests, capacity, n_classes, fares, lp_bl)
        lp_rev[run] = r["revenue"]
        lp_sold[run] = r["seats_sold"]
        lp_spill[run] = r["spill"]

    return {
        "FCFS": PolicyMetrics("FCFS", fcfs_rev, fcfs_sold, fcfs_spill, capacity),
        "EMSR-b": PolicyMetrics("EMSR-b", emsr_rev, emsr_sold, emsr_spill, capacity),
        "LP": PolicyMetrics("LP", lp_rev, lp_sold, lp_spill, capacity),
    }
