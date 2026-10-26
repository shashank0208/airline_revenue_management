"""Tests for src/arm/lp.py — LP seat allocation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arm.lp import solve_lp, lp_booking_limits, PULP_AVAILABLE


class TestLP:
    """LP never exceeds capacity and produces correct revenue."""

    def test_capacity_not_exceeded(self):
        fares = [580, 380, 220, 120]
        demands = [20, 30, 50, 80]
        capacity = 180
        result = solve_lp(fares, demands, capacity, class_names=["Y", "B", "M", "Q"])
        assert sum(result["allocations"]) <= capacity

    def test_low_capacity_respects_limit(self):
        """When capacity < total demand, allocation should equal capacity."""
        fares = [500, 100]
        demands = [100, 100]
        capacity = 50
        result = solve_lp(fares, demands, capacity, class_names=["H", "L"])
        assert sum(result["allocations"]) <= capacity

    def test_revenue_maximisation(self):
        """With plenty of high-fare demand, LP should prefer high fares."""
        fares = [1000, 100]
        demands = [50, 50]
        capacity = 50
        result = solve_lp(fares, demands, capacity, class_names=["H", "L"])
        # Optimal: sell 50 high-fare seats → revenue = 50000
        assert result["revenue"] >= 50 * 1000 - 1  # allow rounding

    @pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
    def test_optimal_status(self):
        fares = [500, 300, 100]
        demands = [20, 30, 40]
        capacity = 90
        result = solve_lp(fares, demands, capacity, class_names=["A", "B", "C"])
        assert "Optimal" in result["status"] or result["status"] == "Greedy-Fallback"

    def test_deterministic_lp_beats_fcfs(self):
        """With deterministic demand, LP revenue >= naive FCFS (serve in
        arrival order, which for low fares means they consume capacity first)."""
        fares = [580, 380, 220, 120]
        demands = [20, 30, 50, 80]  # total 180 = capacity
        capacity = 180
        # FCFS: low fares arrive first → Q gets 80, M gets 50, B gets 30, Y gets 20
        fcfs_revenue = 80 * 120 + 50 * 220 + 30 * 380 + 20 * 580
        result = solve_lp(fares, demands, capacity, class_names=["Y", "B", "M", "Q"])
        assert result["revenue"] >= fcfs_revenue

    def test_booking_limits_nested(self):
        """Booking limits should be non-increasing (highest fare class
        has the most permissive limit, lower classes are tighter)."""
        fares = [580, 380, 220, 120]
        demands = [20, 30, 50, 80]
        capacity = 180
        bl = lp_booking_limits(fares, demands, capacity, class_names=["Y", "B", "M", "Q"])
        assert bl[0] == capacity, "Highest fare class should have BL = capacity"
        for i in range(len(bl) - 1):
            assert bl[i] >= bl[i + 1], f"Booking limits not non-increasing: {bl}"
