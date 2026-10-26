"""
End-to-end pipeline for the Airline Revenue Management Simulator.

Usage:
    python run_pipeline.py [--config configs/default.yaml]

Steps:
    1. Generate synthetic demand data
    2. Train forecasting models (ARIMA + Prophet)
    3. Compute EMSR-b protection levels
    4. Solve LP allocation
    5. Run Monte Carlo simulations (500 runs per route)
    6. Report results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from arm.data import generate_demand, load_config, split_train_test
from arm.forecasting import forecast_all, select_best_models, forecast_summary_df
from arm.emsr import compute_emsr_b
from arm.lp import solve_lp
from arm.simulate import run_simulation
from arm.metrics import compute_paired_comparison


def main(config_path: str = "configs/default.yaml") -> dict:
    """Run full pipeline. Returns results dict for use by Streamlit."""
    config = load_config(config_path)
    seed = config.get("random_seed", 42)

    print("=" * 60)
    print("  Airline Revenue Management Simulator — Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Generate synthetic demand
    # ------------------------------------------------------------------
    print("\n[1/6] Generating synthetic demand data...")
    demand_df = generate_demand(config, seed=seed)
    holdout = config["demand"]["holdout_days"]
    train_df, test_df = split_train_test(demand_df, holdout_days=holdout)
    print(f"  Total records: {len(demand_df)}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # ------------------------------------------------------------------
    # 2. Forecasting
    # ------------------------------------------------------------------
    print("\n[2/6] Training forecasting models (ARIMA)...")
    forecast_results = forecast_all(train_df, test_df, config)
    best_models = select_best_models(forecast_results)
    summary = forecast_summary_df(forecast_results)
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. EMSR-b protection levels
    # ------------------------------------------------------------------
    print("\n[3/6] Computing EMSR-b protection levels...")
    routes = [r["name"] for r in config["routes"]]
    fare_classes = config["fare_classes"]
    # Fare classes ordered high → low for EMSR-b
    fc_sorted = sorted(fare_classes, key=lambda x: x["fare"], reverse=True)
    fc_codes_sorted = [fc["code"] for fc in fc_sorted]
    fares_sorted = [fc["fare"] for fc in fc_sorted]

    emsr_results = {}
    for route in routes:
        # Compute demand stats from training data
        means = []
        stds = []
        for fc in fc_sorted:
            mask = (train_df["route"] == route) & (train_df["fare_class"] == fc["code"])
            series = train_df.loc[mask, "demand"]
            means.append(float(series.mean()))
            stds.append(float(series.std()))

        capacity = next(r["capacity"] for r in config["routes"] if r["name"] == route)
        emsr = compute_emsr_b(fares_sorted, means, stds, capacity)
        emsr_results[route] = emsr
        print(f"  {route}: Protection = {emsr['protection_levels']}, BL = {emsr['booking_limits']}")

    # ------------------------------------------------------------------
    # 4. LP allocation
    # ------------------------------------------------------------------
    print("\n[4/6] Solving LP seat allocations...")
    lp_results = {}
    for route in routes:
        means = []
        for fc in fc_sorted:
            mask = (train_df["route"] == route) & (train_df["fare_class"] == fc["code"])
            means.append(float(train_df.loc[mask, "demand"].mean()))

        capacity = next(r["capacity"] for r in config["routes"] if r["name"] == route)
        # Use total horizon demand = daily_mean * booking_horizon
        horizon = config.get("booking_horizon_days", 90)
        demand_forecasts = [m * horizon for m in means]
        result = solve_lp(fares_sorted, demand_forecasts, capacity, class_names=fc_codes_sorted)
        lp_results[route] = result
        print(f"  {route}: Alloc = {dict(zip(fc_codes_sorted, result['allocations']))}, "
              f"Revenue = ${result['revenue']:,.0f}, Status = {result['status']}")

    # ------------------------------------------------------------------
    # 5. Monte Carlo simulation
    # ------------------------------------------------------------------
    num_runs = config["simulation"]["num_runs"]
    print(f"\n[5/6] Running Monte Carlo simulation ({num_runs} runs per route)...")

    sim_results = {}
    for route in routes:
        means = []
        stds = []
        for fc in fc_sorted:
            mask = (train_df["route"] == route) & (train_df["fare_class"] == fc["code"])
            series = train_df.loc[mask, "demand"]
            # Scale to total demand over booking horizon
            horizon = config.get("booking_horizon_days", 90)
            means.append(float(series.mean()) * horizon)
            stds.append(float(series.std()) * np.sqrt(horizon))

        capacity = next(r["capacity"] for r in config["routes"] if r["name"] == route)

        metrics = run_simulation(
            config=config,
            route_name=route,
            demand_means=means,
            demand_stds=stds,
            fares=fares_sorted,
            fare_class_codes=fc_codes_sorted,
            capacity=capacity,
            num_runs=num_runs,
            base_seed=seed,
        )
        sim_results[route] = metrics

    # ------------------------------------------------------------------
    # 6. Report results
    # ------------------------------------------------------------------
    print("\n[6/6] Results Summary")
    print("=" * 60)

    all_comparisons = {}
    for route in routes:
        metrics = sim_results[route]
        print(f"\n--- {route} ---")
        for policy_name in ["FCFS", "EMSR-b", "LP"]:
            pm = metrics[policy_name]
            ci = pm.ci_95
            print(f"  {policy_name:8s}: Mean Rev = ${pm.mean_revenue:,.0f}  "
                  f"95% CI = [${ci[0]:,.0f}, ${ci[1]:,.0f}]  "
                  f"Load Factor = {pm.mean_load_factor:.1%}  "
                  f"Spill = {pm.mean_spill:.1f}")

        # Paired comparisons vs FCFS
        fcfs = metrics["FCFS"]
        comparisons = {}
        for policy_name in ["EMSR-b", "LP"]:
            pc = compute_paired_comparison(metrics[policy_name], fcfs)
            comparisons[policy_name] = pc
            print(f"  {policy_name} vs FCFS: Uplift = {pc.mean_uplift_pct:+.2f}%, "
                  f"Win Rate = {pc.win_rate:.1%}, "
                  f"p-value (t-test) = {pc.p_value_ttest:.4f}")
        all_comparisons[route] = comparisons

    # Save results to JSON for Streamlit
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True)

    # Save demand data
    demand_df.to_csv(results_dir / "demand_data.csv", index=False)

    # Save forecast summary
    summary.to_csv(results_dir / "forecast_summary.csv", index=False)

    # Save simulation results
    sim_summary = []
    for route in routes:
        for policy_name in ["FCFS", "EMSR-b", "LP"]:
            pm = sim_results[route][policy_name]
            ci = pm.ci_95
            row = {
                "route": route,
                "policy": policy_name,
                "mean_revenue": round(pm.mean_revenue, 2),
                "ci_lower": round(ci[0], 2),
                "ci_upper": round(ci[1], 2),
                "mean_load_factor": round(pm.mean_load_factor, 4),
                "mean_spill": round(pm.mean_spill, 2),
            }
            sim_summary.append(row)

    pd.DataFrame(sim_summary).to_csv(results_dir / "simulation_summary.csv", index=False)

    # Save paired comparisons
    comp_summary = []
    for route in routes:
        for policy_name in ["EMSR-b", "LP"]:
            pc = all_comparisons[route][policy_name]
            comp_summary.append({
                "route": route,
                "policy": policy_name,
                "uplift_pct": round(pc.mean_uplift_pct, 4),
                "win_rate": round(pc.win_rate, 4),
                "p_value_ttest": round(pc.p_value_ttest, 6),
                "p_value_wilcoxon": round(pc.p_value_wilcoxon, 6),
            })
    pd.DataFrame(comp_summary).to_csv(results_dir / "paired_comparisons.csv", index=False)

    # Save EMSR results
    emsr_summary = []
    for route in routes:
        emsr = emsr_results[route]
        for i, code in enumerate(fc_codes_sorted):
            emsr_summary.append({
                "route": route,
                "fare_class": code,
                "fare": fares_sorted[i],
                "booking_limit": emsr["booking_limits"][i],
                "protection_level": emsr["protection_levels"][i - 1] if i > 0 else None,
            })
    pd.DataFrame(emsr_summary).to_csv(results_dir / "emsr_results.csv", index=False)

    # Save LP results
    lp_summary = []
    for route in routes:
        result = lp_results[route]
        for i, code in enumerate(fc_codes_sorted):
            lp_summary.append({
                "route": route,
                "fare_class": code,
                "fare": fares_sorted[i],
                "allocation": result["allocations"][i],
            })
    pd.DataFrame(lp_summary).to_csv(results_dir / "lp_results.csv", index=False)

    # Save raw revenues for distribution plots
    for route in routes:
        for policy_name in ["FCFS", "EMSR-b", "LP"]:
            pm = sim_results[route][policy_name]
            np.save(results_dir / f"{route}_{policy_name}_revenues.npy", pm.revenues)

    print(f"\nResults saved to {results_dir}/")
    print("Done!")

    return {
        "demand_df": demand_df,
        "train_df": train_df,
        "test_df": test_df,
        "forecast_results": forecast_results,
        "best_models": best_models,
        "forecast_summary": summary,
        "emsr_results": emsr_results,
        "lp_results": lp_results,
        "sim_results": sim_results,
        "comparisons": all_comparisons,
        "config": config,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARM pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
