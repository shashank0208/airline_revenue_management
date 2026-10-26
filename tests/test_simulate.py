"""Tests for src/arm/simulate.py — booking simulation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arm.simulate import generate_request_stream, simulate_fcfs, simulate_emsr, run_simulation
from arm.data import load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


class TestRequestStream:
    def test_reproducibility(self, config):
        """Same seed → same stream."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        profile = config["booking_time_profile"]
        codes = ["Y", "B", "M", "Q"]
        demands = [15, 25, 40, 55]

        s1 = generate_request_stream(demands, 90, profile, codes, rng1)
        s2 = generate_request_stream(demands, 90, profile, codes, rng2)
        assert s1 == s2

    def test_requests_within_horizon(self, config):
        """All request days should be in [0, horizon)."""
        rng = np.random.default_rng(99)
        profile = config["booking_time_profile"]
        codes = ["Y", "B", "M", "Q"]
        demands = [15, 25, 40, 55]
        horizon = 90

        stream = generate_request_stream(demands, horizon, profile, codes, rng)
        for day, cls in stream:
            assert 0 <= day < horizon
            assert 0 <= cls < 4


class TestFCFS:
    def test_never_exceeds_capacity(self):
        """FCFS should never sell more than capacity."""
        requests = [(d, c % 4) for d in range(90) for c in range(5)]
        fares = [580, 380, 220, 120]
        capacity = 10
        result = simulate_fcfs(requests, capacity, 4, fares)
        assert result["seats_sold"] <= capacity

    def test_revenue_non_negative(self):
        requests = [(0, 0), (1, 1), (2, 2)]
        result = simulate_fcfs(requests, 180, 4, [580, 380, 220, 120])
        assert result["revenue"] >= 0


class TestEMSR:
    def test_never_exceeds_capacity(self):
        """EMSR should never sell more than capacity."""
        requests = [(d, c % 4) for d in range(90) for c in range(5)]
        fares = [580, 380, 220, 120]
        capacity = 10
        bl = [10, 8, 5, 3]
        result = simulate_emsr(requests, capacity, 4, fares, bl)
        assert result["seats_sold"] <= capacity
