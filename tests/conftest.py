"""Shared fixtures: a synthetic-data config that runs offline and finishes fast."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Make `ai_trader` importable without `pip install -e .` when running pytest from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def synthetic_config():
    """Minimal config using synthetic OHLCV — no network, no AAPL download."""
    return {
        "device": "cpu",
        "env": {
            "data_source": "synthetic",
            "synthetic_length": 600,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "initial_cash": 10_000,
            "transaction_cost": 0.0003,
            "slippage": 0.0001,
            "max_position": 5,
            "allow_short": False,
            "trade_size": 1,
            "lookback_window": 10,
            "random_start": False,
            "reward_scale": 100.0,
            "risk_penalty": 0.001,
            "position_penalty": 0.0002,
            "inactivity_penalty": 0.005,
        },
        "agent": {
            "gamma": 0.99,
            "learning_rate": 0.0005,
            "batch_size": 8,
            "buffer_size": 200,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.99,
            "tau": 0.01,
        },
        "training": {
            "seed": 42,
            "episodes": 2,
            "max_steps_per_episode": 30,
            "eval_every": 0,
            "eval_episodes": 1,
        },
    }
