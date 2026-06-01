"""Interactive Streamlit dashboard for inspecting a trained Double DQN agent.

Run with: streamlit run scripts/demo_app.py
"""

import _bootstrap  # noqa: F401

from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ai_trader.agents import build_agent
from ai_trader.env import build_data_bundle, make_env
from ai_trader.training import buy_and_hold_curve, summarize_episode
from ai_trader.utils import load_config, set_seed
from ai_trader.viz import (
    BG,
    C_CYAN,
    C_GREEN,
    C_ORANGE,
    C_RED,
    C_WHITE,
    GRID,
    plotly_dark_layout,
)


st.set_page_config(
    page_title="AI Trader — Double DQN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stMetric { background: #1a1f2b; padding: 12px; border-radius: 8px; border: 1px solid #3a3f4b; }
    .stMetric label { color: #9da5b4 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e6eaf0 !important; }
    div[data-testid="stSidebar"] { background: #0e1117; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seed(int(cfg["training"]["seed"]))
    bundle = build_data_bundle(cfg)
    return cfg, bundle


def load_agent(cfg, bundle, checkpoint_path: str):
    env = make_env(cfg, split="test", data_bundle=bundle)
    agent = build_agent(
        cfg,
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
    )
    agent.load(checkpoint_path)
    return agent, env


@st.cache_data
def precompute_episode(
    _agent, _env, max_steps: int, seed: int, initial_cash: float
) -> Dict[str, Any]:
    """Roll the full episode once and cache step-by-step series for slider scrubbing."""
    obs, info = _env.reset(seed=seed)

    prices = [info["price"]]
    equity = [initial_cash]
    cash_hist = [initial_cash]
    position_hist = [0]
    drawdown_hist = [0.0]
    actions = []

    for _ in range(max_steps):
        action = _agent.choose_action(obs, explore=False)
        obs, _, terminated, truncated, info = _env.step(action)
        prices.append(info["price"])
        equity.append(info["portfolio_value"])
        cash_hist.append(info["cash"])
        position_hist.append(info["position"])
        drawdown_hist.append(info["drawdown"] * 100)
        actions.append(action)
        if terminated or truncated:
            break

    return {
        "prices": prices,
        "equity": equity,
        "cash": cash_hist,
        "positions": position_hist,
        "drawdowns": drawdown_hist,
        "actions": actions,
    }


def find_checkpoints() -> Dict[str, str]:
    ckpts: Dict[str, str] = {}
    candidates = [Path("results"), Path("runs")]
    for d in candidates:
        if d.exists():
            for f in d.rglob("*_best.pt"):
                ckpts[str(f)] = str(f)
            for f in d.rglob("*_latest.pt"):
                ckpts[str(f)] = str(f)
    return ckpts


# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("📈 AI Trader")
st.sidebar.markdown("**Double DQN Trading Agent**")

page = st.sidebar.radio(
    "Navigate",
    [
        "🎬 Live Agent Replay",
        "🏁 Agent vs Buy & Hold",
        "🎛️ Hyperparameter Explorer",
    ],
)

ckpt_map = find_checkpoints()
cfg, bundle = load_resources()


# ════════════════════════════════════════════════════════════════════════
# PAGE 1: Live Agent Replay
# ════════════════════════════════════════════════════════════════════════
if page == "🎬 Live Agent Replay":
    st.title("Live Agent Replay")
    st.caption(
        "Scrub through the agent's trading decisions on unseen test data. "
        "All data is pre-computed for smooth viewing."
    )

    ckpt_path = st.sidebar.selectbox(
        "Checkpoint", list(ckpt_map.keys()) or ["No checkpoints found"]
    )
    demo_seed = st.sidebar.number_input("Seed", value=42, step=1)

    if ckpt_path and ckpt_path != "No checkpoints found":
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
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            height=700,
            margin=dict(l=60, r=30, t=40, b=40),
            legend=dict(bgcolor="rgba(26,31,43,0.8)", bordercolor="#3a3f4b",
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
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
    else:
        st.warning("No checkpoints found. Train an agent first with `python scripts/train.py`.")


# ════════════════════════════════════════════════════════════════════════
# PAGE 2: Agent vs Buy & Hold Race
# ════════════════════════════════════════════════════════════════════════
elif page == "🏁 Agent vs Buy & Hold":
    st.title("Agent vs Buy & Hold Race")
    st.caption("Compare the DDQN agent's equity curve against a passive buy-and-hold strategy.")

    ckpt_path = st.sidebar.selectbox(
        "Checkpoint", list(ckpt_map.keys()) or ["No checkpoints found"]
    )
    demo_seed = st.sidebar.number_input("Seed", value=42, step=1)

    if ckpt_path and ckpt_path != "No checkpoints found":
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
            height=500,
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Trading Day",
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
    else:
        st.warning("No checkpoints found.")


# ════════════════════════════════════════════════════════════════════════
# PAGE 3: Hyperparameter Explorer
# ════════════════════════════════════════════════════════════════════════
elif page == "🎛️ Hyperparameter Explorer":
    st.title("Hyperparameter Explorer")
    st.caption("Interactively explore how key hyperparameters affect agent behavior.")

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

    st.subheader("Epsilon Decay Schedule")
    decay_rate = st.slider("Decay rate", 0.990, 0.999, 0.995, 0.001)
    eps_min = st.slider("Epsilon min", 0.01, 0.20, 0.05, 0.01)
    episodes = 5000

    epsilons = []
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
        xaxis_title="Episode", yaxis_title="Epsilon"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

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
        xaxis_title="Trading Day", yaxis_title="Cumulative Return (%)"))
    st.plotly_chart(fig, use_container_width=True)
