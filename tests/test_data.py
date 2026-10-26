"""Tests for src/arm/data.py — synthetic demand generator."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arm.data import generate_demand, load_config, split_train_test


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


class TestGenerateDemand:
    """Seed-based reproducibility and basic sanity checks."""

    def test_reproducibility(self, config):
        """Same seed → identical DataFrames."""
        df1 = generate_demand(config, seed=42)
        df2 = generate_demand(config, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_differs(self, config):
        """Different seeds → different data."""
        df1 = generate_demand(config, seed=42)
        df2 = generate_demand(config, seed=99)
        assert not df1["demand"].equals(df2["demand"])

    def test_columns(self, config):
        df = generate_demand(config, seed=42)
        assert set(df.columns) == {"date", "route", "fare_class", "fare", "demand"}

    def test_no_negative_demand(self, config):
        df = generate_demand(config, seed=42)
        assert (df["demand"] >= 0).all()

    def test_expected_shape(self, config):
        """5 routes × 4 classes × 365 days = 7300 rows."""
        df = generate_demand(config, seed=42)
        n_routes = len(config["routes"])
        n_classes = len(config["fare_classes"])
        n_days = config["demand"]["history_days"]
        assert len(df) == n_routes * n_classes * n_days

    def test_routes_present(self, config):
        df = generate_demand(config, seed=42)
        expected_routes = {r["name"] for r in config["routes"]}
        assert set(df["route"].unique()) == expected_routes

    def test_fare_classes_present(self, config):
        df = generate_demand(config, seed=42)
        expected = {fc["code"] for fc in config["fare_classes"]}
        assert set(df["fare_class"].unique()) == expected


class TestSplitTrainTest:
    def test_split_sizes(self, config):
        df = generate_demand(config, seed=42)
        holdout = config["demand"]["holdout_days"]
        train, test = split_train_test(df, holdout_days=holdout)
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == len(df)

    def test_no_date_overlap(self, config):
        df = generate_demand(config, seed=42)
        train, test = split_train_test(df, holdout_days=30)
        assert train["date"].max() < test["date"].min()
