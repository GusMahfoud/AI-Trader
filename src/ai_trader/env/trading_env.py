"""Gymnasium trading environment with portfolio accounting and reward shaping."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("gymnasium is required. Install with: pip install gymnasium") from exc

from ai_trader.data import DataBundle, build_data_bundle


class TradingEnv(gym.Env):
    """Discrete-action trading env over a chronologically split price series.

    Actions: 0 = hold, 1 = sell, 2 = buy. Reward is the scaled per-step
    portfolio return minus risk / position / inactivity penalties.
    """

    metadata = {"render_modes": []}

    def __init__(self, data_bundle: DataBundle, config: Dict[str, Any], split: str = "train"):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self._df = data_bundle.frame.reset_index(drop=True)
        self._feature_columns = data_bundle.feature_columns
        self._split_bounds = data_bundle.split_bounds
        self._split = split

        # Pre-extract to numpy so step() avoids pandas iloc on every tick.
        self._prices_arr: np.ndarray = self._df["close"].values.astype(np.float32)
        self._dates_arr: np.ndarray = self._df["date"].values
        self._features_arr: np.ndarray = (
            self._df[self._feature_columns].values.astype(np.float32)
        )

        self._cfg = config.get("env", {})
        self.lookback_window = int(self._cfg.get("lookback_window", 20))
        self.initial_cash = float(self._cfg.get("initial_cash", 10_000.0))
        self.transaction_cost = float(self._cfg.get("transaction_cost", 0.001))
        self.slippage = float(self._cfg.get("slippage", 0.0005))
        self.max_position = int(self._cfg.get("max_position", 10))
        self.allow_short = bool(self._cfg.get("allow_short", False))
        self.trade_size = int(self._cfg.get("trade_size", 1))
        self.reward_scale = float(self._cfg.get("reward_scale", 1.0))
        self.risk_penalty = float(self._cfg.get("risk_penalty", 0.0))
        self.position_penalty = float(self._cfg.get("position_penalty", 0.0))
        self.inactivity_penalty = float(self._cfg.get("inactivity_penalty", 0.0))
        self.random_start = bool(self._cfg.get("random_start", split == "train"))
        self._hold_streak: int = 0

        raw_sl = self._cfg.get("stop_loss", None)
        raw_tp = self._cfg.get("take_profit", None)
        self.stop_loss: Optional[float] = float(raw_sl) if raw_sl is not None else None
        self.take_profit: Optional[float] = float(raw_tp) if raw_tp is not None else None

        self._min_position = -self.max_position if self.allow_short else 0
        self._episode_steps_cap = int(config.get("training", {}).get("max_steps_per_episode", 252))

        # State = lookback window of market features + 4 portfolio features.
        obs_dim = self.lookback_window * len(self._feature_columns) + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._rng = np.random.default_rng(int(config.get("training", {}).get("seed", 42)))

        self._episode_start: int = 0
        self._cursor: int = 0
        self._end_idx: int = 0
        self._step_count: int = 0

        self._cash: float = self.initial_cash
        self._position: int = 0
        self._portfolio_value: float = self.initial_cash
        self._peak_value: float = self.initial_cash

        self._equity_curve: List[float] = []
        self._position_curve: List[int] = []
        self._action_history: List[int] = []
        self._price_history: List[float] = []
        self._date_history: List[pd.Timestamp] = []

    def _get_split_limits(self) -> Tuple[int, int]:
        lo, hi = self._split_bounds[self._split]
        lo = max(lo, self.lookback_window)
        hi = min(hi, len(self._df) - 1)
        if hi - lo < self.lookback_window + 5:
            raise ValueError(
                f"Not enough rows in split='{self._split}' for lookback_window={self.lookback_window}."
            )
        return lo, hi

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        lo, hi = self._get_split_limits()
        max_len = min(self._episode_steps_cap, hi - lo - 1)
        if max_len < 5:
            raise ValueError("Episode length too short. Reduce lookback_window or increase data length.")

        if self.random_start:
            start_max = hi - max_len
            self._episode_start = int(self._rng.integers(lo, max(start_max + 1, lo + 1)))
        else:
            self._episode_start = lo

        self._cursor = self._episode_start
        self._end_idx = min(self._cursor + max_len, hi)
        self._step_count = 0

        self._cash = self.initial_cash
        self._position = 0
        self._portfolio_value = self.initial_cash
        self._peak_value = self.initial_cash

        self._equity_curve = [self._portfolio_value]
        self._position_curve = [self._position]
        self._action_history = []
        self._hold_streak = 0

        price = self._price_at(self._cursor)
        date = self._date_at(self._cursor)
        self._price_history = [price]
        self._date_history = [date]

        return self._get_state(), self._info(action_masked=False, full=True)

    def _price_at(self, idx: int) -> float:
        return float(self._prices_arr[idx])

    def _date_at(self, idx: int) -> pd.Timestamp:
        return pd.Timestamp(self._dates_arr[idx])

    def _get_state(self) -> np.ndarray:
        start = self._cursor - self.lookback_window + 1
        market_flat = self._features_arr[start : self._cursor + 1].reshape(-1)

        price = self._price_at(self._cursor)
        exposure = (self._position * price) / (abs(self._portfolio_value) + 1e-8)
        cash_frac = self._cash / (abs(self._portfolio_value) + 1e-8)
        unrealized = (self._portfolio_value - self.initial_cash) / (self.initial_cash + 1e-8)
        pos_frac = self._position / max(1, self.max_position)

        portfolio_vec = np.array([pos_frac, cash_frac, exposure, unrealized], dtype=np.float32)
        return np.concatenate([market_flat, portfolio_vec], axis=0).astype(np.float32)

    def _apply_action(self, action: int, price: float) -> Tuple[int, bool, float]:
        action = int(action)
        masked = False
        trade_units = 0

        if action == 2:  # buy
            if self._position + self.trade_size <= self.max_position:
                exec_price = price * (1.0 + self.slippage)
                cost = exec_price * self.trade_size * (1.0 + self.transaction_cost)
                if self._cash >= cost:
                    self._cash -= cost
                    self._position += self.trade_size
                    trade_units = self.trade_size
                else:
                    masked = True
            else:
                masked = True
        elif action == 1:  # sell
            if self._position - self.trade_size >= self._min_position:
                exec_price = price * (1.0 - self.slippage)
                proceeds = exec_price * self.trade_size * (1.0 - self.transaction_cost)
                self._cash += proceeds
                self._position -= self.trade_size
                trade_units = -self.trade_size
            else:
                masked = True

        return action, masked, float(trade_units)

    def _info(self, action_masked: bool, trade_units: float = 0.0, full: bool = False) -> Dict[str, Any]:
        drawdown = max(0.0, (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-8))
        info: Dict[str, Any] = {
            "date": str(self._date_history[-1]),
            "price": self._price_history[-1],
            "cash": self._cash,
            "position": self._position,
            "portfolio_value": self._portfolio_value,
            "drawdown": drawdown,
            "action_masked": bool(action_masked),
            "trade_units": trade_units,
        }
        # Only copy full history arrays on episode end (or explicit request) to avoid
        # 252 list copies per episode.
        if full:
            info.update(
                {
                    "equity_curve": list(self._equity_curve),
                    "position_curve": list(self._position_curve),
                    "action_history": list(self._action_history),
                    "price_history": list(self._price_history),
                    "date_history": [str(d) for d in self._date_history],
                }
            )
        return info

    def _forced_exit_action(self, action: int, price: float) -> int:
        """Override action to sell if stop-loss or take-profit thresholds are hit."""
        if self._position <= 0:
            return action
        current_value = self._cash + self._position * price
        drawdown = (self._peak_value - current_value) / (self._peak_value + 1e-8)
        gain = (current_value - self.initial_cash) / (self.initial_cash + 1e-8)
        if self.stop_loss is not None and drawdown >= self.stop_loss:
            return 1
        if self.take_profit is not None and gain >= self.take_profit:
            return 1
        return action

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._price_at(self._cursor)
        prev_value = self._cash + self._position * current_price

        action = self._forced_exit_action(action, current_price)
        action, masked, trade_units = self._apply_action(action=action, price=current_price)
        self._action_history.append(int(action))

        self._cursor += 1
        self._step_count += 1

        next_price = self._price_at(self._cursor)
        self._portfolio_value = self._cash + self._position * next_price
        self._peak_value = max(self._peak_value, self._portfolio_value)

        step_return = (self._portfolio_value - prev_value) / (abs(prev_value) + 1e-8)
        drawdown = max(0.0, (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-8))
        risk_cost = self.risk_penalty * drawdown
        inventory_cost = self.position_penalty * abs(self._position) / max(1, self.max_position)

        # Penalty for sitting idle with no position — prevents convergence to all-hold.
        if action == 0 and self._position == 0:
            self._hold_streak += 1
        else:
            self._hold_streak = 0
        inactivity_cost = self.inactivity_penalty * min(self._hold_streak, 10) / 10.0

        reward = float(self.reward_scale * step_return - risk_cost - inventory_cost - inactivity_cost)

        self._equity_curve.append(self._portfolio_value)
        self._position_curve.append(self._position)
        self._price_history.append(next_price)
        self._date_history.append(self._date_at(self._cursor))

        terminated = bool(self._cursor >= self._end_idx or self._portfolio_value <= 0.0)
        truncated = False

        return (
            self._get_state(),
            reward,
            terminated,
            truncated,
            self._info(masked, trade_units, full=terminated),
        )

    def close(self) -> None:
        return


def make_env(
    config: Dict[str, Any],
    split: str = "train",
    data_bundle: Optional[DataBundle] = None,
) -> TradingEnv:
    bundle = data_bundle if data_bundle is not None else build_data_bundle(config)
    return TradingEnv(data_bundle=bundle, config=config, split=split)


def make_env_bundle(
    config: Dict[str, Any],
) -> Tuple[TradingEnv, TradingEnv, TradingEnv]:
    bundle = build_data_bundle(config)
    train_env = make_env(config, split="train", data_bundle=bundle)
    val_env = make_env(config, split="val", data_bundle=bundle)
    test_env = make_env(config, split="test", data_bundle=bundle)
    return train_env, val_env, test_env
