"""Page 3: Hyperparameter Explorer — interactive epsilon, decay, and transaction cost widgets."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ai_trader.viz import C_CYAN, C_GREEN, C_ORANGE, C_RED, C_WHITE, plotly_dark_layout


def render() -> None:
    st.title("Hyperparameter Explorer")
    st.caption("Interactively explore how key hyperparameters affect agent behavior.")

    # ── Epsilon-Greedy Exploration ───────────────────────────────────────
    st.subheader("Epsilon-Greedy Exploration")
    st.markdown("Adjust epsilon to see how the explore/exploit balance changes.")

    eps = st.slider("Epsilon (ε)", 0.0, 1.0, 0.1, 0.01)
    n_decisions = 1000
    np.random.seed(42)
    explore_count = int(np.sum(np.random.rand(n_decisions) < eps))
    exploit_count = n_decisions - explore_count

    c1, c2 = st.columns(2)
    c1.metric("Explore (random)", f"{explore_count}/{n_decisions}",
              delta=f"{explore_count / n_decisions * 100:.1f}%")
    c2.metric("Exploit (best Q)", f"{exploit_count}/{n_decisions}",
              delta=f"{exploit_count / n_decisions * 100:.1f}%")

    fig = go.Figure(go.Pie(
        labels=["Explore", "Exploit"],
        values=[explore_count, exploit_count],
        marker=dict(colors=[C_ORANGE, C_CYAN]),
        hole=0.5,
        textinfo="percent+label",
    ))
    fig.update_layout(**plotly_dark_layout(title=f"Action Selection at ε = {eps:.2f}", height=350))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Epsilon Decay Schedule ───────────────────────────────────────────
    st.subheader("Epsilon Decay Schedule")
    decay_rate = st.slider("Decay rate", 0.990, 0.999, 0.995, 0.001)
    eps_min = st.slider("Epsilon min", 0.01, 0.20, 0.05, 0.01)
    episodes = 5000

    epsilons: list[float] = []
    e = 1.0
    for _ in range(episodes):
        epsilons.append(e)
        e = max(eps_min, e * decay_rate)

    reach_min = next((i for i, v in enumerate(epsilons) if v <= eps_min + 0.001), episodes)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=epsilons, mode="lines", name="Epsilon",
                             line=dict(color=C_CYAN, width=2.5)))
    fig.add_hline(y=eps_min, line_dash="dot", line_color=C_RED,
                  annotation_text=f"ε_min = {eps_min}")
    fig.add_vline(x=reach_min, line_dash="dot", line_color=C_GREEN,
                  annotation_text=f"Reaches min at ep {reach_min}")
    fig.update_layout(**plotly_dark_layout(
        title=f"Epsilon Decay (rate={decay_rate})", height=350,
        xaxis_title="Episode", yaxis_title="Epsilon",
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Transaction Cost Impact ──────────────────────────────────────────
    st.subheader("Transaction Cost Impact")
    st.markdown("See how different transaction costs affect round-trip trading profitability.")
    tc = st.slider("Transaction cost (%)", 0.01, 2.0, 0.1, 0.01)
    daily_return = st.slider("Expected daily return (%)", 0.01, 1.0, 0.15, 0.01)

    round_trip_cost = tc * 2
    net_per_trade = daily_return - round_trip_cost
    trades_needed = int(np.ceil(round_trip_cost / max(daily_return, 0.001)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Round-trip cost", f"{round_trip_cost:.2f}%")
    c2.metric("Net per trade", f"{net_per_trade:+.2f}%",
              delta="Profitable" if net_per_trade > 0 else "Unprofitable")
    c3.metric("Days to break even", f"{trades_needed}")

    days = 252
    profits_no_cost = [0.0]
    profits_with_cost = [0.0]
    for _ in range(1, days + 1):
        profits_no_cost.append(profits_no_cost[-1] + daily_return)
        profits_with_cost.append(profits_with_cost[-1] + net_per_trade)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=profits_no_cost, mode="lines", name="No costs",
                             line=dict(color=C_GREEN, width=2)))
    fig.add_trace(go.Scatter(y=profits_with_cost, mode="lines",
                             name=f"With {tc:.2f}% cost",
                             line=dict(color=C_RED, width=2)))
    fig.add_hline(y=0, line_color=C_WHITE, line_dash="dot")
    fig.update_layout(**plotly_dark_layout(
        title="Cumulative P&L Over 252 Trading Days", height=350,
        xaxis_title="Trading Day", yaxis_title="Cumulative Return (%)",
    ))
    st.plotly_chart(fig, use_container_width=True)
