"""Evaluation helpers: summarize_episode metrics + buy_and_hold curve."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trader.training.evaluate import (
    buy_and_hold_curve,
    evaluate_buy_and_hold_policy,
    evaluate_random_policy,
    summarize_episode,
)
from ai_trader.env import make_env_bundle


def test_summarize_empty_equity_returns_safe_defaults():
    out = summarize_episode({}, initial_cash=10_000.0)
    assert out["final_value"] == 10_000.0
    assert out["total_return"] == 0.0
    assert out["num_trades"] == 0.0


def test_summarize_constant_equity_has_zero_return_and_dd():
    info = {
        "equity_curve": [10_000.0] * 5,
        "action_history": [0, 0, 0, 0],
    }
    out = summarize_episode(info, initial_cash=10_000.0)
    assert out["total_return"] == 0.0
    assert out["max_drawdown"] == 0.0
    assert out["num_trades"] == 0.0


def test_summarize_counts_buys_and_sells_as_trades():
    info = {
        "equity_curve": [10_000.0, 10_100.0],
        "action_history": [0, 1, 2, 2, 0],
    }
    out = summarize_episode(info, initial_cash=10_000.0)
    assert out["num_trades"] == 3.0  # one sell + two buys
    assert out["total_return"] > 0


def test_summarize_drawdown_is_negative_after_peak_then_fall():
    info = {
        "equity_curve": [10_000.0, 12_000.0, 11_000.0, 9_000.0],
        "action_history": [0, 0, 0],
    }
    out = summarize_episode(info, initial_cash=10_000.0)
    # Peak = 12_000, trough = 9_000 → -25% drawdown.
    assert out["max_drawdown"] < 0
    np.testing.assert_allclose(out["max_drawdown"], -0.25, rtol=1e-6)


def test_buy_and_hold_curve_starts_at_initial_cash():
    prices = [100.0, 105.0, 110.0, 102.0]
    curve = buy_and_hold_curve(prices, initial_cash=10_000.0)
    assert len(curve) == len(prices)
    np.testing.assert_allclose(curve[0], 10_000.0, rtol=1e-6)


def test_buy_and_hold_curve_tracks_price_proportionally():
    prices = [100.0, 200.0]
    curve = buy_and_hold_curve(prices, initial_cash=10_000.0)
    np.testing.assert_allclose(curve[1], 20_000.0, rtol=1e-6)


def test_random_policy_returns_expected_keys(synthetic_config):
    _, _, test_env = make_env_bundle(synthetic_config)
    metrics = evaluate_random_policy(test_env, episodes=1, max_steps=20, seed=42)
    for key in (
        "avg_reward",
        "std_reward",
        "avg_total_return",
        "avg_final_value",
        "avg_max_drawdown",
        "avg_sharpe",
        "avg_num_trades",
    ):
        assert key in metrics


def test_buy_and_hold_policy_returns_expected_keys(synthetic_config):
    _, _, test_env = make_env_bundle(synthetic_config)
    metrics = evaluate_buy_and_hold_policy(test_env, episodes=1, max_steps=20, seed=42)
    for key in (
        "avg_reward",
        "std_reward",
        "avg_total_return",
        "avg_final_value",
        "avg_max_drawdown",
        "avg_sharpe",
        "avg_num_trades",
    ):
        assert key in metrics
    test_env.close()


def test_buy_and_hold_ramps_to_max_position_then_holds(synthetic_config):
    _, _, test_env = make_env_bundle(synthetic_config)
    max_position = synthetic_config["env"]["max_position"]
    metrics = evaluate_buy_and_hold_policy(test_env, episodes=1, max_steps=20, seed=42)
    # B&H buys one unit per step until full (max_position), then holds — so it
    # makes exactly max_position trades over the episode.
    assert metrics["avg_num_trades"] == pytest.approx(float(max_position), abs=0.01)
    test_env.close()


def test_buy_and_hold_fraction_mode_deploys_then_holds(synthetic_config):
    cfg = dict(synthetic_config)
    cfg["env"] = {
        **cfg["env"], "position_sizing": "fraction", "trade_fraction": 0.25, "max_exposure": 1.0,
    }
    _, _, test_env = make_env_bundle(cfg)
    metrics = evaluate_buy_and_hold_policy(test_env, episodes=1, max_steps=30, seed=42)
    assert "avg_sharpe" in metrics
    # ~4 buys (0.25 steps) to reach full exposure, then holds — a handful of trades, not 0.
    assert 1 <= metrics["avg_num_trades"] <= 10
    test_env.close()
