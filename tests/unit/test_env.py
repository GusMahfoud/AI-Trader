"""Trading env: reset/step contract, observation shape, episode termination."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trader.env import build_data_bundle, make_env, make_env_bundle


def test_reset_returns_correct_observation_shape(synthetic_config):
    env = make_env(synthetic_config, split="train")
    obs, info = env.reset(seed=42)

    n_features = len(env._feature_columns)
    expected_dim = env.lookback_window * n_features + 4
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)

    # Reset always populates full history fields.
    assert "equity_curve" in info
    assert "action_history" in info


def test_step_returns_valid_tuple(synthetic_config):
    env = make_env(synthetic_config, split="train")
    obs, _ = env.reset(seed=42)

    obs_next, reward, terminated, truncated, info = env.step(2)  # buy
    assert obs_next.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "portfolio_value" in info


def test_buy_decreases_cash_increases_position(synthetic_config):
    env = make_env(synthetic_config, split="train")
    env.reset(seed=42)
    cash_before, pos_before = env._cash, env._position
    env.step(2)  # buy
    assert env._position == pos_before + env.trade_size
    assert env._cash < cash_before


def test_sell_with_zero_long_position_is_masked(synthetic_config):
    """Selling from a flat long-only book is rejected — action is masked, no trade fires."""
    cfg = synthetic_config
    env = make_env(cfg, split="train")
    env.reset(seed=42)

    cash_before, pos_before = env._cash, env._position
    _, _, _, _, info = env.step(1)  # sell
    assert info["action_masked"] is True
    assert env._position == pos_before
    assert env._cash == cash_before


def test_episode_terminates_within_max_steps(synthetic_config):
    env = make_env(synthetic_config, split="val")
    env.reset(seed=42)
    max_steps = synthetic_config["training"]["max_steps_per_episode"]

    for _ in range(max_steps + 10):
        _, _, terminated, _, _ = env.step(0)
        if terminated:
            break
    assert terminated, "Env did not terminate within expected step cap"


def test_make_env_bundle_returns_three_envs(synthetic_config):
    envs = make_env_bundle(synthetic_config)
    assert len(envs) == 3
    train_env, val_env, test_env = envs
    assert train_env._split == "train"
    assert val_env._split == "val"
    assert test_env._split == "test"


def test_data_bundle_has_no_nan_features(synthetic_config):
    bundle = build_data_bundle(synthetic_config)
    arr = bundle.frame[bundle.feature_columns].values
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_chronological_split_bounds_are_disjoint(synthetic_config):
    bundle = build_data_bundle(synthetic_config)
    train_lo, train_hi = bundle.split_bounds["train"]
    val_lo, val_hi = bundle.split_bounds["val"]
    test_lo, test_hi = bundle.split_bounds["test"]
    assert train_hi == val_lo
    assert val_hi == test_lo
    assert train_lo < train_hi < val_hi < test_hi


def _fraction_cfg(synthetic_config, **overrides):
    cfg = dict(synthetic_config)
    cfg["env"] = {
        **cfg["env"],
        "position_sizing": "fraction",
        "trade_fraction": 0.25,
        "max_exposure": 1.0,
        **overrides,
    }
    return cfg


def test_fraction_mode_deploys_real_capital(synthetic_config):
    """Fraction sizing should invest most of equity — the old shares cap left ~80% idle."""
    env = make_env(_fraction_cfg(synthetic_config), split="train")
    env.reset(seed=42)
    for _ in range(8):
        env.step(2)  # buy
    price = env._price_at(env._cursor)
    exposure = env._position * price / env._portfolio_value
    assert exposure > 0.5


def test_fraction_mode_respects_max_exposure(synthetic_config):
    env = make_env(_fraction_cfg(synthetic_config, trade_fraction=0.5, max_exposure=0.5), split="train")
    env.reset(seed=42)
    for _ in range(12):
        env.step(2)  # keep buying — must not exceed the exposure cap
    price = env._price_at(env._cursor)
    exposure = env._position * price / env._portfolio_value
    assert exposure <= 0.6  # ~0.5 cap plus rounding/price drift


def test_fraction_mode_pos_frac_stays_bounded(synthetic_config):
    env = make_env(_fraction_cfg(synthetic_config), split="train")
    env.reset(seed=42)
    for _ in range(8):
        obs, *_ = env.step(2)
    # State layout: [...market features..., pos_frac, cash_frac, exposure, unrealized].
    pos_frac = float(obs[-4])
    assert -1.5 <= pos_frac <= 1.5


@pytest.mark.parametrize("seed", [0, 7, 99])
def test_reset_is_seed_deterministic(synthetic_config, seed):
    cfg = dict(synthetic_config)
    cfg["env"] = {**cfg["env"], "random_start": True}
    env_a = make_env(cfg, split="train")
    env_b = make_env(cfg, split="train")
    obs_a, _ = env_a.reset(seed=seed)
    obs_b, _ = env_b.reset(seed=seed)
    np.testing.assert_array_equal(obs_a, obs_b)
