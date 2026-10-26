"""Tests for src/arm/emsr.py — EMSR-b protection levels."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arm.emsr import emsr_b_protection_levels, protection_to_booking_limits, compute_emsr_b


class TestEMSRb:
    """Protection level correctness and monotonicity."""

    def test_monotonic_protection_levels(self):
        """Protection levels should be non-decreasing as we aggregate
        more high-fare classes (y_1 <= y_2 <= … <= y_{n-1})."""
        fares = [580, 380, 220, 120]  # Y, B, M, Q (high → low)
        means = [15, 25, 40, 55]
        stds = [5, 8, 10, 12]
        pl = emsr_b_protection_levels(fares, means, stds)
        for i in range(len(pl) - 1):
            assert pl[i] <= pl[i + 1], f"Protection not monotonic: {pl}"

    def test_two_class_simple(self):
        """Two classes: protection level should be positive when high fare
        is meaningfully above low fare."""
        fares = [500, 100]
        means = [30, 60]
        stds = [8, 12]
        pl = emsr_b_protection_levels(fares, means, stds)
        assert len(pl) == 1
        assert pl[0] > 0  # should protect seats for higher fare

    def test_booking_limits_sum(self):
        """Booking limits should be <= capacity and class-0 limit == capacity."""
        fares = [580, 380, 220, 120]
        means = [15, 25, 40, 55]
        stds = [5, 8, 10, 12]
        capacity = 180
        result = compute_emsr_b(fares, means, stds, capacity)
        bl = result["booking_limits"]
        assert bl[0] == capacity
        for lim in bl:
            assert 0 <= lim <= capacity

    def test_protection_levels_non_negative(self):
        """All protection levels must be >= 0."""
        fares = [580, 380, 220, 120]
        means = [10, 20, 30, 40]
        stds = [3, 5, 7, 9]
        pl = emsr_b_protection_levels(fares, means, stds)
        for p in pl:
            assert p >= 0

    def test_empty_classes(self):
        """Single class → no protection needed."""
        pl = emsr_b_protection_levels([500], [50], [10])
        assert pl == []
