"""Cached data loading and episode computation helpers shared across dashboard pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from ai_trader.agents import build_agent
from ai_trader.env import build_data_bundle, make_env
from ai_trader.utils import load_config, set_seed


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
    for d in [Path("results"), Path("runs")]:
        if d.exists():
            for f in d.rglob("*_best.pt"):
                ckpts[str(f)] = str(f)
            for f in d.rglob("*_latest.pt"):
                ckpts[str(f)] = str(f)
    return ckpts
