"""
Streamlit dashboard for the Airline Revenue Management Simulator.

Pages:
    1. Data & Forecasts — demand data, forecast vs actual charts
    2. Optimizer — EMSR-b protection levels, LP allocations
    3. Simulation — Monte Carlo results, revenue distributions, uplift
    4. Assumptions — model documentation

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arm.data import generate_demand, load_config, split_train_test, load_csv_demand
from arm.forecasting import forecast_all, select_best_models, forecast_summary_df
from arm.emsr import compute_emsr_b
from arm.lp import solve_lp
from arm.simulate import run_simulation
from arm.metrics import compute_paired_comparison

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Airline Revenue Management Simulator",
    page_icon="✈️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("✈️ ARM Simulator")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Data & Forecasts", "⚙️ Optimizer", "🎲 Simulation", "📋 Assumptions"],
)

# Config
config_path = ROOT / "configs" / "default.yaml"
config = load_config(config_path)

# Allow CSV upload
uploaded_file = st.sidebar.file_uploader("Upload demand CSV (optional)", type=["csv"])

# Simulation controls
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
num_runs = st.sidebar.slider("Monte Carlo runs", 50, 1000, config["simulation"]["num_runs"], step=50)
seed = st.sidebar.number_input("Random seed", value=config.get("random_seed", 42), step=1)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def get_demand_data(_config, _seed, csv_bytes=None):
    if csv_bytes is not None:
        import io
        return load_csv_demand(io.BytesIO(csv_bytes))
    return generate_demand(_config, seed=_seed)


csv_bytes = uploaded_file.read() if uploaded_file else None
demand_df = get_demand_data(config, seed, csv_bytes)
holdout = config["demand"]["holdout_days"]
train_df, test_df = split_train_test(demand_df, holdout_days=holdout)

# Sorted fare classes (high → low)
fare_classes = config["fare_classes"]
fc_sorted = sorted(fare_classes, key=lambda x: x["fare"], reverse=True)
fc_codes_sorted = [fc["code"] for fc in fc_sorted]
fares_sorted = [fc["fare"] for fc in fc_sorted]
routes = [r["name"] for r in config["routes"]]


# =========================================================================
# PAGE 1: Data & Forecasts
# =========================================================================
if page == "📊 Data & Forecasts":
    st.title("📊 Demand Data & Forecasts")

    # --- Data overview ---
    st.header("Synthetic Demand Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Routes", len(routes))
    col2.metric("Fare Classes", len(fare_classes))
    col3.metric("Total Records", f"{len(demand_df):,}")

    # Route selector
    selected_route = st.selectbox("Select route", routes)

    # Demand over time
    route_data = demand_df[demand_df["route"] == selected_route]
    if PLOTLY_AVAILABLE:
        fig = px.line(
            route_data,
            x="date",
            y="demand",
            color="fare_class",
            title=f"Daily Demand — {selected_route}",
            labels={"demand": "Bookings", "date": "Date"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(f"Daily Demand — {selected_route}")
        pivoted = route_data.pivot(index="date", columns="fare_class", values="demand")
        st.line_chart(pivoted)

    # --- Forecasting ---
    st.header("Forecast Evaluation")

    with st.spinner("Training forecast models (ARIMA)..."):
        forecast_results = forecast_all(train_df, test_df, config)
        best_models = select_best_models(forecast_results)
        summary = forecast_summary_df(forecast_results)

    st.dataframe(summary, use_container_width=True)

    # Forecast vs actual chart
    st.subheader("Forecast vs Actual")
    fc_select = st.selectbox("Fare class", fc_codes_sorted, key="fc_forecast")

    key = (selected_route, fc_select)
    if key in best_models:
        best = best_models[key]
        test_dates = test_df[
            (test_df["route"] == selected_route) & (test_df["fare_class"] == fc_select)
        ].sort_values("date")["date"].values

        if PLOTLY_AVAILABLE:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=test_dates, y=best.actuals, mode="lines+markers",
                                       name="Actual", line=dict(color="#636EFA")))
            fig2.add_trace(go.Scatter(x=test_dates, y=best.predictions, mode="lines+markers",
                                       name=f"{best.model_name} Forecast", line=dict(color="#EF553B", dash="dash")))
            fig2.update_layout(
                title=f"{selected_route} / {fc_select} — {best.model_name} (MAPE: {best.mape:.1f}%)",
                xaxis_title="Date", yaxis_title="Demand",
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption(f"{selected_route} / {fc_select} — {best.model_name} (MAPE: {best.mape:.1f}%)")
            df_chart = pd.DataFrame(
                {"Actual": best.actuals, f"{best.model_name} Forecast": best.predictions},
                index=test_dates
            )
            st.line_chart(df_chart)

# =========================================================================
# PAGE 2: Optimizer
# =========================================================================
elif page == "⚙️ Optimizer":
    st.title("⚙️ Optimizer Outputs")

    selected_route = st.selectbox("Select route", routes, key="opt_route")
    capacity = next(r["capacity"] for r in config["routes"] if r["name"] == selected_route)

    # Compute demand stats
    means, stds = [], []
    for fc in fc_sorted:
        mask = (train_df["route"] == selected_route) & (train_df["fare_class"] == fc["code"])
        series = train_df.loc[mask, "demand"]
        means.append(float(series.mean()))
        stds.append(float(series.std()))

    # --- EMSR-b ---
    st.header("EMSR-b Protection Levels")
    emsr = compute_emsr_b(fares_sorted, means, stds, capacity)

    emsr_df = pd.DataFrame({
        "Fare Class": fc_codes_sorted,
        "Fare ($)": fares_sorted,
        "Mean Demand": [round(m, 1) for m in means],
        "Std Dev": [round(s, 1) for s in stds],
        "Protection Level": ["-"] + [str(p) for p in emsr["protection_levels"]],
        "Booking Limit": emsr["booking_limits"],
    })
    st.dataframe(emsr_df, use_container_width=True)

    # Protection levels bar chart
    if PLOTLY_AVAILABLE:
        fig_emsr = go.Figure()
        fig_emsr.add_trace(go.Bar(
            x=fc_codes_sorted, y=emsr["booking_limits"],
            marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
            text=emsr["booking_limits"], textposition="auto",
        ))
        fig_emsr.update_layout(
            title=f"EMSR-b Booking Limits — {selected_route} (Capacity: {capacity})",
            xaxis_title="Fare Class", yaxis_title="Booking Limit",
            height=400,
        )
        fig_emsr.add_hline(y=capacity, line_dash="dash", line_color="red",
                            annotation_text=f"Capacity = {capacity}")
        st.plotly_chart(fig_emsr, use_container_width=True)
    else:
        st.subheader(f"EMSR-b Booking Limits — {selected_route} (Capacity: {capacity})")
        df_bl = pd.DataFrame({"Booking Limit": emsr["booking_limits"]}, index=fc_codes_sorted)
        st.bar_chart(df_bl)

    # --- LP ---
    st.header("LP Seat Allocation")
    horizon = config.get("booking_horizon_days", 90)
    demand_forecasts = [m * horizon for m in means]
    lp_result = solve_lp(fares_sorted, demand_forecasts, capacity, class_names=fc_codes_sorted)

    lp_df = pd.DataFrame({
        "Fare Class": fc_codes_sorted,
        "Fare ($)": fares_sorted,
        "Forecast Demand": [round(d, 0) for d in demand_forecasts],
        "LP Allocation": lp_result["allocations"],
        "Revenue Contribution ($)": [f * a for f, a in zip(fares_sorted, lp_result["allocations"])],
    })
    st.dataframe(lp_df, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Total LP Revenue", f"${lp_result['revenue']:,.0f}")
    col2.metric("LP Status", lp_result["status"])

    # LP allocation chart
    if PLOTLY_AVAILABLE:
        fig_lp = px.pie(
            lp_df, values="LP Allocation", names="Fare Class",
            title=f"LP Seat Allocation — {selected_route}",
            color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
        )
        st.plotly_chart(fig_lp, use_container_width=True)
    else:
        st.subheader(f"LP Seat Allocation — {selected_route}")
        st.bar_chart(lp_df.set_index("Fare Class")["LP Allocation"])

# =========================================================================
# PAGE 3: Simulation
# =========================================================================
elif page == "🎲 Simulation":
    st.title("🎲 Monte Carlo Simulation")

    selected_route = st.selectbox("Select route", routes, key="sim_route")
    capacity = next(r["capacity"] for r in config["routes"] if r["name"] == selected_route)

    run_sim = st.button("▶️ Run Simulation", type="primary")

    if run_sim:
        means, stds = [], []
        for fc in fc_sorted:
            mask = (train_df["route"] == selected_route) & (train_df["fare_class"] == fc["code"])
            series = train_df.loc[mask, "demand"]
            horizon = config.get("booking_horizon_days", 90)
            means.append(float(series.mean()) * horizon)
            stds.append(float(series.std()) * np.sqrt(horizon))

        with st.spinner(f"Running {num_runs} Monte Carlo simulations..."):
            metrics = run_simulation(
                config=config,
                route_name=selected_route,
                demand_means=means,
                demand_stds=stds,
                fares=fares_sorted,
                fare_class_codes=fc_codes_sorted,
                capacity=capacity,
                num_runs=num_runs,
                base_seed=seed,
            )

        st.success(f"Completed {num_runs} runs for {selected_route}")

        # --- Summary table ---
        st.header("Results Summary")
        rows = []
        for policy in ["FCFS", "EMSR-b", "LP"]:
            pm = metrics[policy]
            ci = pm.ci_95
            rows.append({
                "Policy": policy,
                "Mean Revenue ($)": f"{pm.mean_revenue:,.0f}",
                "95% CI Lower ($)": f"{ci[0]:,.0f}",
                "95% CI Upper ($)": f"{ci[1]:,.0f}",
                "Load Factor": f"{pm.mean_load_factor:.1%}",
                "Mean Spill": f"{pm.mean_spill:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # --- Paired comparisons ---
        st.header("Paired Comparisons vs FCFS")
        fcfs = metrics["FCFS"]
        comp_rows = []
        for policy in ["EMSR-b", "LP"]:
            pc = compute_paired_comparison(metrics[policy], fcfs)
            comp_rows.append({
                "Policy": policy,
                "Mean Uplift (%)": f"{pc.mean_uplift_pct:+.2f}%",
                "Win Rate": f"{pc.win_rate:.1%}",
                "p-value (t-test)": f"{pc.p_value_ttest:.4f}",
                "p-value (Wilcoxon)": f"{pc.p_value_wilcoxon:.4f}",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        # --- Revenue distribution ---
        st.header("Revenue Distributions")
        if PLOTLY_AVAILABLE:
            fig_dist = go.Figure()
            colors = {"FCFS": "#636EFA", "EMSR-b": "#EF553B", "LP": "#00CC96"}
            for policy in ["FCFS", "EMSR-b", "LP"]:
                pm = metrics[policy]
                fig_dist.add_trace(go.Histogram(
                    x=pm.revenues, name=policy,
                    marker_color=colors[policy], opacity=0.6,
                    nbinsx=40,
                ))
            fig_dist.update_layout(
                barmode="overlay",
                title=f"Revenue Distribution — {selected_route}",
                xaxis_title="Revenue ($)", yaxis_title="Frequency",
                height=450,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # --- Uplift histogram ---
            st.header("Revenue Uplift vs FCFS")
            fig_uplift = go.Figure()
            for policy in ["EMSR-b", "LP"]:
                diff = metrics[policy].revenues - fcfs.revenues
                fig_uplift.add_trace(go.Histogram(
                    x=diff, name=f"{policy} - FCFS",
                    opacity=0.6, nbinsx=40,
                ))
            fig_uplift.update_layout(
                barmode="overlay",
                title="Per-Run Revenue Uplift vs FCFS",
                xaxis_title="Revenue Uplift ($)", yaxis_title="Frequency",
                height=400,
            )
            fig_uplift.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_uplift, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            colors = {"FCFS": "#1f77b4", "EMSR-b": "#d62728", "LP": "#2ca02c"}
            for policy in ["FCFS", "EMSR-b", "LP"]:
                ax.hist(metrics[policy].revenues, bins=35, alpha=0.55, label=policy, color=colors[policy])
            ax.set_title(f"Revenue Distribution — {selected_route}")
            ax.set_xlabel("Revenue ($)")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

            st.header("Revenue Uplift vs FCFS")
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            for policy in ["EMSR-b", "LP"]:
                diff = metrics[policy].revenues - fcfs.revenues
                ax2.hist(diff, bins=35, alpha=0.55, label=f"{policy} - FCFS")
            ax2.axvline(0, color="black", linestyle="--")
            ax2.set_title("Per-Run Revenue Uplift vs FCFS")
            ax2.set_xlabel("Revenue Uplift ($)")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            st.pyplot(fig2)

    else:
        # Check for pre-computed results
        results_dir = ROOT / "results"
        if (results_dir / "simulation_summary.csv").exists():
            st.info("Showing pre-computed results. Click 'Run Simulation' for fresh results.")
            sim_df = pd.read_csv(results_dir / "simulation_summary.csv")
            route_sim = sim_df[sim_df["route"] == selected_route]
            st.dataframe(route_sim, use_container_width=True)

            if (results_dir / "paired_comparisons.csv").exists():
                comp_df = pd.read_csv(results_dir / "paired_comparisons.csv")
                route_comp = comp_df[comp_df["route"] == selected_route]
                st.dataframe(route_comp, use_container_width=True)

            # Precomputed distribution plots
            loaded_revs = {}
            for policy in ["FCFS", "EMSR-b", "LP"]:
                path = results_dir / f"{selected_route}_{policy}_revenues.npy"
                if path.exists():
                    loaded_revs[policy] = np.load(path)

            if len(loaded_revs) == 3:
                st.header("Pre-computed Revenue Distributions")
                if PLOTLY_AVAILABLE:
                    fig_dist = go.Figure()
                    colors = {"FCFS": "#636EFA", "EMSR-b": "#EF553B", "LP": "#00CC96"}
                    for policy, revs in loaded_revs.items():
                        fig_dist.add_trace(go.Histogram(
                            x=revs, name=policy,
                            marker_color=colors[policy], opacity=0.6,
                            nbinsx=40,
                        ))
                    fig_dist.update_layout(
                        barmode="overlay",
                        title=f"Revenue Distribution — {selected_route}",
                        xaxis_title="Revenue ($)", yaxis_title="Frequency",
                        height=450,
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    colors = {"FCFS": "#1f77b4", "EMSR-b": "#d62728", "LP": "#2ca02c"}
                    for policy, revs in loaded_revs.items():
                        ax.hist(revs, bins=35, alpha=0.55, label=policy, color=colors[policy])
                    ax.set_title(f"Revenue Distribution — {selected_route}")
                    ax.set_xlabel("Revenue ($)")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    st.pyplot(fig)
        else:
            st.info("Click 'Run Simulation' to start, or run `python run_pipeline.py` first.")

# =========================================================================
# PAGE 4: Assumptions
# =========================================================================
elif page == "📋 Assumptions":
    st.title("📋 Model Assumptions & Documentation")

    st.markdown("""
    ## Demand Generation
    - **5 synthetic routes** with varying demand multipliers
    - **4 fare classes**: Q ($120) < M ($220) < B ($380) < Y ($580)
    - Daily demand = base × route_multiplier × trend × weekly_seasonality × holiday_spike + noise
    - **Trend**: small linear growth (+0.1% per day)
    - **Weekly seasonality**: sine wave with period 7 days, amplitude 20%
    - **Holiday spikes**: ~6 dates/year with 1.8× multiplier
    - **Noise**: Gaussian, σ = 25% of base demand, clipped to ≥ 0

    ## Booking Time Profile
    - Low fares (Q) book early, high fares (Y) book late
    - Modelled with Beta distribution CDF over the 90-day booking horizon
    - Q: Beta(2, 5), M: Beta(2.5, 4), B: Beta(3.5, 3), Y: Beta(5, 2)

    ## Forecasting
    - **ARIMA**: SARIMAX(1,1,1)(1,1,1,7) — weekly seasonal component
    - **Prophet**: weekly seasonality enabled, yearly disabled (only 1 year of data)
    - Holdout: last 30 days; metrics: MAPE, MAE, sMAPE
    - Best model per (route, fare_class) selected by lowest MAPE

    ## EMSR-b
    - Nested protection levels assuming normally distributed aggregate demand
    - Protection level y_j: P(D_{1..j} > y_j) = f_{j+1} / f̄_j
    - f̄_j = demand-weighted average fare of classes 1..j (highest fares)

    ## Linear Program
    - Maximise Σ fare_i × x_i subject to Σ x_i ≤ capacity, 0 ≤ x_i ≤ demand_i
    - Solved with PuLP (CBC solver)
    - Booking limits derived as cumulative allocations (nested)

    ## Simulation
    - Poisson arrivals modulated by Beta CDF time profile
    - Common random numbers: all policies see the same request stream per run
    - 500 runs per route (configurable)

    ## Limitations
    1. **Synthetic data only** — no real airline booking data
    2. **Single-leg model** — no network/connection effects
    3. **No cancellations, no-shows, or overbooking**
    4. **Static booking limits** — EMSR-b and LP limits computed once, not re-optimised
    5. **Normal demand assumption** — real demand may be skewed
    6. **No competitor pricing** effects
    7. **Simplified customer behaviour** — no buy-up/buy-down across fare classes
    """)

    st.markdown("---")
    st.subheader("Configuration")
    st.json(config)
