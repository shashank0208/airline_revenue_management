"""Tests for src/arm/metrics.py — metrics and paired comparisons."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arm.metrics import PolicyMetrics, compute_paired_comparison


class TestPolicyMetrics:
    def test_mean_revenue(self):
        pm = PolicyMetrics("test", np.array([100, 200, 300]), np.array([1, 1, 1]), np.array([0, 0, 0]), 180)
        assert pm.mean_revenue == 200.0

    def test_load_factor(self):
        pm = PolicyMetrics("test", np.array([100.0]), np.array([90.0]), np.array([0.0]), 180)
        assert abs(pm.mean_load_factor - 0.5) < 1e-9

    def test_ci_contains_mean(self):
        rng = np.random.default_rng(42)
        revs = rng.normal(1000, 50, size=500)
        pm = PolicyMetrics("test", revs, np.full(500, 100.0), np.zeros(500), 180)
        lo, hi = pm.ci_95
        assert lo < pm.mean_revenue < hi


class TestPairedComparison:
    def test_uplift_positive_when_better(self):
        base = PolicyMetrics("FCFS", np.array([100.0, 100, 100]), np.array([1, 1, 1]), np.zeros(3), 180)
        better = PolicyMetrics("LP", np.array([120.0, 120, 120]), np.array([1, 1, 1]), np.zeros(3), 180)
        pc = compute_paired_comparison(better, base)
        assert pc.mean_uplift_pct > 0
        assert pc.win_rate == 1.0

    def test_uplift_zero_when_equal(self):
        a = PolicyMetrics("A", np.array([100.0, 100, 100]), np.array([1, 1, 1]), np.zeros(3), 180)
        b = PolicyMetrics("B", np.array([100.0, 100, 100]), np.array([1, 1, 1]), np.zeros(3), 180)
        pc = compute_paired_comparison(a, b)
        assert abs(pc.mean_uplift_pct) < 1e-9
