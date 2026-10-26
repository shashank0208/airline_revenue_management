"""
Demand forecasting with ARIMA and (optionally) Prophet.

For each (route, fare_class) series the module:
1. Fits both ARIMA and Prophet on the training set.
2. Forecasts the holdout period.
3. Computes MAPE, MAE, sMAPE.
4. Picks the model with lowest MAPE.

If Prophet is not installed the module logs a warning and uses ARIMA only.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

# Try importing Prophet — it's optional
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. Only ARIMA will be used.")


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Ignores zeros in actual."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    denom = (np.abs(a) + np.abs(p))
    mask = denom != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(2 * np.abs(a[mask] - p[mask]) / denom[mask]) * 100)


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Container for one model's forecast output."""
    model_name: str
    route: str
    fare_class: str
    predictions: np.ndarray
    actuals: np.ndarray
    mape: float
    mae: float
    smape: float
    fitted_model: object = field(repr=False, default=None)


def _fit_arima(
    train_series: pd.Series,
    forecast_steps: int,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
) -> np.ndarray:
    """Fit SARIMAX and return point forecast for `forecast_steps` ahead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
    return result.forecast(steps=forecast_steps).values


def _fit_prophet(
    train_df: pd.DataFrame,
    forecast_steps: int,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = False,
) -> np.ndarray:
    """Fit Prophet and return point forecast."""
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed.")

    df = train_df.rename(columns={"date": "ds", "demand": "y"})
    m = Prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
    )
    m.fit(df)
    future = m.make_future_dataframe(periods=forecast_steps)
    fc = m.predict(future)
    preds = fc["yhat"].iloc[-forecast_steps:].values
    return np.clip(preds, 0, None)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def forecast_all(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> list[ForecastResult]:
    """
    Forecast every (route, fare_class) combination.

    Returns a list of ForecastResult, two per combo (ARIMA + Prophet) when
    Prophet is available, otherwise one (ARIMA only).
    """
    cfg_fc = config.get("forecasting", {})
    arima_order = tuple(cfg_fc.get("arima_order", [1, 1, 1]))
    arima_seasonal = tuple(cfg_fc.get("arima_seasonal_order", [1, 1, 1, 7]))
    prophet_weekly = cfg_fc.get("prophet_weekly_seasonality", True)
    prophet_yearly = cfg_fc.get("prophet_yearly_seasonality", False)

    holdout_days = config["demand"]["holdout_days"]

    results: list[ForecastResult] = []

    groups = train_df.groupby(["route", "fare_class"])
    test_groups = test_df.groupby(["route", "fare_class"])

    for (route, fc_code), grp in groups:
        # Prepare series
        ts = grp.sort_values("date").set_index("date")["demand"]
        test_grp = test_groups.get_group((route, fc_code)).sort_values("date")
        actuals = test_grp["demand"].values
        n_steps = len(actuals)

        # --- ARIMA ---
        try:
            arima_preds = _fit_arima(ts, n_steps, order=arima_order, seasonal_order=arima_seasonal)
            arima_preds = np.clip(arima_preds, 0, None)
        except Exception as e:
            logger.warning("ARIMA failed for %s/%s: %s", route, fc_code, e)
            arima_preds = np.full(n_steps, ts.mean())

        results.append(
            ForecastResult(
                model_name="ARIMA",
                route=route,
                fare_class=fc_code,
                predictions=arima_preds,
                actuals=actuals,
                mape=mape(actuals, arima_preds),
                mae=mae(actuals, arima_preds),
                smape=smape(actuals, arima_preds),
            )
        )

        # --- Prophet ---
        if PROPHET_AVAILABLE:
            try:
                prophet_preds = _fit_prophet(
                    grp[["date", "demand"]],
                    n_steps,
                    weekly_seasonality=prophet_weekly,
                    yearly_seasonality=prophet_yearly,
                )
            except Exception as e:
                logger.warning("Prophet failed for %s/%s: %s", route, fc_code, e)
                prophet_preds = np.full(n_steps, ts.mean())

            results.append(
                ForecastResult(
                    model_name="Prophet",
                    route=route,
                    fare_class=fc_code,
                    predictions=prophet_preds,
                    actuals=actuals,
                    mape=mape(actuals, prophet_preds),
                    mae=mae(actuals, prophet_preds),
                    smape=smape(actuals, prophet_preds),
                )
            )

    return results


def select_best_models(results: list[ForecastResult]) -> dict[tuple[str, str], ForecastResult]:
    """
    For each (route, fare_class), pick the model with the lowest MAPE.

    Returns dict keyed by (route, fare_class).
    """
    best: dict[tuple[str, str], ForecastResult] = {}
    for r in results:
        key = (r.route, r.fare_class)
        if key not in best or r.mape < best[key].mape:
            best[key] = r
    return best


def forecast_summary_df(results: list[ForecastResult]) -> pd.DataFrame:
    """Return a tidy DataFrame summarising all forecast results."""
    rows = []
    for r in results:
        rows.append(
            {
                "route": r.route,
                "fare_class": r.fare_class,
                "model": r.model_name,
                "MAPE": round(r.mape, 2),
                "MAE": round(r.mae, 2),
                "sMAPE": round(r.smape, 2),
            }
        )
    return pd.DataFrame(rows)
