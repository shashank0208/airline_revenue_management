"""Tests for src/arm/forecasting.py — accuracy metrics and forecast logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arm.forecasting import mape, mae, smape


class TestMAPE:
    """MAPE on known tiny examples."""

    def test_perfect_forecast(self):
        actual = np.array([10.0, 20.0, 30.0])
        assert mape(actual, actual) == 0.0

    def test_known_value(self):
        actual = np.array([100.0, 200.0])
        pred = np.array([110.0, 180.0])
        # errors: 10/100=0.10, 20/200=0.10 → mean = 0.10 → 10%
        assert abs(mape(actual, pred) - 10.0) < 1e-9

    def test_ignores_zero_actuals(self):
        actual = np.array([0.0, 100.0])
        pred = np.array([5.0, 110.0])
        # Only the second element counts: 10/100 = 10%
        assert abs(mape(actual, pred) - 10.0) < 1e-9


class TestMAE:
    def test_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == 0.0

    def test_known(self):
        a = np.array([10.0, 20.0])
        p = np.array([12.0, 18.0])
        assert abs(mae(a, p) - 2.0) < 1e-9


class TestSMAPE:
    def test_perfect(self):
        a = np.array([10.0, 20.0])
        assert smape(a, a) == 0.0

    def test_known(self):
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 180.0])
        # Element 1: 2*10 / (100+110) = 20/210
        # Element 2: 2*20 / (200+180) = 40/380
        expected = ((20 / 210) + (40 / 380)) / 2 * 100
        assert abs(smape(a, p) - expected) < 1e-6


class TestForecastPipeline:
    def test_fit_arima_runs_and_returns_correct_length(self):
        import pandas as pd
        from arm.forecasting import _fit_arima

        # Simple synthetic series with 30 days
        dates = pd.date_range("2025-01-01", periods=30)
        s = pd.Series(np.linspace(10, 20, 30) + np.random.RandomState(42).normal(0, 1, 30), index=dates)
        preds = _fit_arima(s, forecast_steps=7, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
        assert len(preds) == 7
        assert not np.isnan(preds).any()

    def test_forecast_all_and_selection(self):
        import pandas as pd
        from arm.forecasting import forecast_all, select_best_models, forecast_summary_df

        dates_train = pd.date_range("2025-01-01", periods=20)
        dates_test = pd.date_range("2025-01-21", periods=5)

        train_rows = []
        test_rows = []
        for d in dates_train:
            train_rows.append({"route": "JFK-LAX", "fare_class": "Y", "date": d, "demand": 15.0})
        for d in dates_test:
            test_rows.append({"route": "JFK-LAX", "fare_class": "Y", "date": d, "demand": 16.0})

        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)
        config = {
            "demand": {"holdout_days": 5},
            "forecasting": {
                "arima_order": [1, 0, 0],
                "arima_seasonal_order": [0, 0, 0, 0],
            },
        }

        results = forecast_all(train_df, test_df, config)
        assert len(results) >= 1
        assert results[0].route == "JFK-LAX"
        assert len(results[0].predictions) == 5

        best = select_best_models(results)
        assert ("JFK-LAX", "Y") in best

        summary = forecast_summary_df(results)
        assert not summary.empty
        assert "MAPE" in summary.columns
