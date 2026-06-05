"""Page 2: Agent vs Buy & Hold Race — filled-area equity curve comparison."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ai_trader.viz import C_CYAN, C_ORANGE, plotly_dark_layout

from app.helpers import load_agent, precompute_episode


def render(cfg: Dict[str, Any], bundle, ckpt_map: Dict[str, str]) -> None:
    st.title("Agent vs Buy & Hold Race")
    st.caption("Compare the DDQN agent's equity curve against a passive buy-and-hold strategy.")

    ckpt_path = st.sidebar.selectbox(
        "Checkpoint", list(ckpt_map.keys()) or ["No checkpoints found"]
    )
    demo_seed = st.sidebar.number_input("Seed", value=42, step=1)

    if not ckpt_path or ckpt_path == "No checkpoints found":
        st.warning("No checkpoints found.")
        return

    agent, env = load_agent(cfg, bundle, ckpt_path)
    max_steps = int(cfg["training"]["max_steps_per_episode"])
    initial_cash = env.initial_cash

    data = precompute_episode(agent, env, max_steps, int(demo_seed), initial_cash)
    total_steps = len(data["actions"])
    step = st.slider("Trading Day", 1, total_steps, total_steps, 1)

    vis_prices = data["prices"][: step + 1]
    vis_equity = data["equity"][: step + 1]
    bh_shares = initial_cash / max(data["prices"][0], 1e-8)
    vis_bh = [bh_shares * p for p in vis_prices]

    x = list(range(len(vis_equity)))
    agent_arr = np.array(vis_equity)
    bh_arr = np.array(vis_bh)
    winning = agent_arr[-1] >= bh_arr[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(agent_arr) + list(bh_arr[::-1]),
        fill="toself",
        fillcolor=f"rgba({'0,217,126' if winning else '231,76,60'},0.12)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=x, y=list(agent_arr), mode="lines",
                             name="Double DQN", line=dict(color=C_CYAN, width=3)))
    fig.add_trace(go.Scatter(x=x, y=list(bh_arr), mode="lines",
                             name="Buy & Hold", line=dict(color=C_ORANGE, width=3)))
    fig.add_annotation(x=x[-1], y=float(agent_arr[-1]), text=f"${agent_arr[-1]:,.0f}",
                       showarrow=False, xanchor="left", xshift=10,
                       font=dict(color=C_CYAN, size=14, family="monospace"))
    fig.add_annotation(x=x[-1], y=float(bh_arr[-1]), text=f"${bh_arr[-1]:,.0f}",
                       showarrow=False, xanchor="left", xshift=10,
                       font=dict(color=C_ORANGE, size=14, family="monospace"))
    fig.update_layout(**plotly_dark_layout(
        title=f"Portfolio Race — Day {step}/{total_steps}",
        height=500, yaxis_title="Portfolio Value ($)", xaxis_title="Trading Day",
    ))
    st.plotly_chart(fig, use_container_width=True)

    pv = float(agent_arr[-1])
    bh_val = float(bh_arr[-1])
    delta = pv - bh_val

    c1, c2, c3 = st.columns(3)
    c1.metric("Double DQN", f"${pv:,.0f}", delta=f"{(pv / initial_cash - 1) * 100:+.1f}%")
    c2.metric("Buy & Hold", f"${bh_val:,.0f}", delta=f"{(bh_val / initial_cash - 1) * 100:+.1f}%")
    c3.metric("Agent Edge", f"${delta:+,.0f}", delta="Winning" if delta >= 0 else "Losing")

    if step == total_steps:
        if delta >= 0:
            st.success(f"🏆 Agent **wins** by ${delta:,.2f}!")
        else:
            st.error(f"Buy & Hold wins by ${abs(delta):,.2f}")
