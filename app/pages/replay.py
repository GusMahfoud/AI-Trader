"""Page 1: Live Agent Replay — scrub through agent decisions on the test split."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ai_trader.viz import BG, C_CYAN, C_GREEN, C_ORANGE, C_RED, C_WHITE, GRID

from app.helpers import load_agent, precompute_episode


def render(cfg: Dict[str, Any], bundle, ckpt_map: Dict[str, str]) -> None:
    st.title("Live Agent Replay")
    st.caption(
        "Scrub through the agent's trading decisions on unseen test data. "
        "All data is pre-computed for smooth viewing."
    )

    ckpt_path = st.sidebar.selectbox(
        "Checkpoint", list(ckpt_map.keys()) or ["No checkpoints found"]
    )
    demo_seed = st.sidebar.number_input("Seed", value=42, step=1)

    if not ckpt_path or ckpt_path == "No checkpoints found":
        st.warning("No checkpoints found. Train an agent first with `python scripts/train.py`.")
        return

    agent, env = load_agent(cfg, bundle, ckpt_path)
    max_steps = int(cfg["training"]["max_steps_per_episode"])
    initial_cash = env.initial_cash

    data = precompute_episode(agent, env, max_steps, int(demo_seed), initial_cash)
    total_steps = len(data["actions"])
    step = st.slider("Trading Day", 1, total_steps, total_steps, 1)

    vis_prices = data["prices"][: step + 1]
    vis_equity = data["equity"][: step + 1]
    vis_actions = data["actions"][:step]

    buy_steps = [i + 1 for i, a in enumerate(vis_actions) if a == 2]
    sell_steps = [i + 1 for i, a in enumerate(vis_actions) if a == 1]
    buy_prices = [data["prices"][s] for s in buy_steps]
    sell_prices = [data["prices"][s] for s in sell_steps]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.35, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("Price & Trades", "Portfolio Value", "Drawdown"),
    )
    x = list(range(len(vis_prices)))

    fig.add_trace(
        go.Scatter(x=x, y=vis_prices, mode="lines", name="Close Price",
                   line=dict(color=C_WHITE, width=2)),
        row=1, col=1,
    )
    if buy_steps:
        fig.add_trace(
            go.Scatter(x=buy_steps, y=buy_prices, mode="markers",
                       name=f"Buy ({len(buy_steps)})",
                       marker=dict(symbol="triangle-up", size=10,
                                   color=C_GREEN, line=dict(width=1, color="white"))),
            row=1, col=1,
        )
    if sell_steps:
        fig.add_trace(
            go.Scatter(x=sell_steps, y=sell_prices, mode="markers",
                       name=f"Sell ({len(sell_steps)})",
                       marker=dict(symbol="triangle-down", size=10,
                                   color=C_RED, line=dict(width=1, color="white"))),
            row=1, col=1,
        )

    bh_curve = [initial_cash / max(data["prices"][0], 1e-8) * p for p in vis_prices]
    fig.add_trace(
        go.Scatter(x=x, y=vis_equity, mode="lines", name="Double DQN",
                   line=dict(color=C_CYAN, width=2.5)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=bh_curve, mode="lines", name="Buy & Hold",
                   line=dict(color=C_ORANGE, width=2, dash="dash")),
        row=2, col=1,
    )

    vis_dd = data["drawdowns"][: step + 1]
    fig.add_trace(
        go.Scatter(x=x, y=[-d for d in vis_dd], mode="lines", name="Drawdown",
                   fill="tozeroy", line=dict(color=C_RED, width=1.5),
                   fillcolor="rgba(231,76,60,0.2)"),
        row=3, col=1,
    )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=700,
        margin=dict(l=60, r=30, t=40, b=40),
        legend=dict(bgcolor="rgba(26,31,43,0.8)", bordercolor="#3a3f4b",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=GRID, showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID, showgrid=True, row=i, col=1)
    fig.update_xaxes(title_text="Trading Day", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="DD (%)", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    pv = data["equity"][step]
    ret = (pv / initial_cash - 1) * 100
    counts = Counter(vis_actions)
    bh_val = bh_curve[-1]
    bh_ret = (bh_val / initial_cash - 1) * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Portfolio", f"${pv:,.0f}")
    c2.metric("Return", f"{ret:+.2f}%")
    c3.metric("vs Buy&Hold", f"{ret - bh_ret:+.2f}%")
    c4.metric("Position", f"{data['positions'][step]} shares")
    c5.metric("Drawdown", f"{data['drawdowns'][step]:.2f}%")
    c6.metric("Trades", f"{counts.get(1, 0) + counts.get(2, 0)}")
