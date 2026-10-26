"""
Synthetic daily demand generator for airline revenue management.

Assumptions (documented here and in the README):
- 5 routes, 4 fare classes (Q < M < B < Y by price).
- Demand is generated per (route, fare_class, date) as a non-negative float
  representing the number of booking requests on that date for the *departure*
  that is `booking_horizon_days` in the future.
- The signal is composed of:
    1. Linear trend  (small positive growth).
    2. Weekly seasonality  (sine wave, period 7).
    3. Holiday spikes  (~6 random dates per year, multiplicative).
    4. Gaussian noise  (clipped to keep demand >= 0).
- Each route has a multiplier so that routes differ in popularity.
- A fixed random seed guarantees reproducibility.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_holidays(year_start: dt.date, num_days: int, rng: np.random.Generator) -> set[dt.date]:
    """Return ~6 deterministic 'holiday' dates spread across the year."""
    # Fixed US-ish holidays relative to year
    base_year = year_start.year
    fixed = [
        dt.date(base_year, 1, 1),    # New Year
        dt.date(base_year, 5, 25),   # Memorial Day (approx)
        dt.date(base_year, 7, 4),    # Independence Day
        dt.date(base_year, 9, 1),    # Labor Day (approx)
        dt.date(base_year, 11, 26),  # Thanksgiving (approx)
        dt.date(base_year, 12, 25),  # Christmas
    ]
    # Also add next year in case history spans >1 year
    next_year = base_year + 1
    fixed += [
        dt.date(next_year, 1, 1),
        dt.date(next_year, 5, 25),
        dt.date(next_year, 7, 4),
        dt.date(next_year, 9, 1),
        dt.date(next_year, 11, 26),
        dt.date(next_year, 12, 25),
    ]
    end_date = year_start + dt.timedelta(days=num_days)
    return {d for d in fixed if year_start <= d < end_date}


def generate_demand(
    config: dict,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic daily demand data.

    Returns a DataFrame with columns:
        date, route, fare_class, fare, demand
    where `demand` is a non-negative float (mean daily bookings).
    """
    seed = seed if seed is not None else config.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    cfg_demand = config["demand"]
    history_days: int = cfg_demand["history_days"]
    base_demand: dict[str, float] = cfg_demand["base_daily_demand"]
    weekly_amp: float = cfg_demand["weekly_amplitude"]
    trend_per_day: float = cfg_demand["trend_per_day"]
    holiday_mult: float = cfg_demand["holiday_multiplier"]
    noise_scale: float = cfg_demand["noise_scale"]
    route_mults: dict[str, float] = cfg_demand["route_multipliers"]

    routes = [r["name"] for r in config["routes"]]
    fare_classes = config["fare_classes"]  # list of dicts

    start_date = dt.date(2024, 1, 1)
    dates = [start_date + dt.timedelta(days=i) for i in range(history_days)]
    holidays = _make_holidays(start_date, history_days, rng)

    records: list[dict] = []

    for route in routes:
        r_mult = route_mults.get(route, 1.0)
        for fc in fare_classes:
            fc_code = fc["code"]
            fare = fc["fare"]
            mu = base_demand[fc_code]

            for i, date in enumerate(dates):
                # 1. Linear trend
                trend = 1.0 + trend_per_day * i

                # 2. Weekly seasonality (peak on Fri=4, trough on Tue=1)
                day_of_week = date.weekday()  # Mon=0 … Sun=6
                weekly = 1.0 + weekly_amp * np.sin(2 * np.pi * (day_of_week - 1) / 7)

                # 3. Holiday spike
                hol = holiday_mult if date in holidays else 1.0

                # Deterministic component
                signal = mu * r_mult * trend * weekly * hol

                # 4. Gaussian noise
                noise = rng.normal(0, noise_scale * mu)
                demand = max(0.0, signal + noise)

                records.append(
                    {
                        "date": date,
                        "route": route,
                        "fare_class": fc_code,
                        "fare": fare,
                        "demand": round(demand, 2),
                    }
                )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_csv_demand(path: str | Path) -> pd.DataFrame:
    """
    Load user-supplied CSV demand data.

    Expected columns: date, route, fare_class, fare, demand
    """
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "route", "fare_class", "fare", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def split_train_test(
    df: pd.DataFrame, holdout_days: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split demand data into train and test sets based on date."""
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test
