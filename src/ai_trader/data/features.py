"""Feature engineering: technical indicators computed from OHLCV data.

Base features are always present. Optional regime features are opt-in via
``env.extra_features`` so they can be ablated cleanly without touching the base set.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

BASE_FEATURES: List[str] = [
    "ret_1",
    "log_ret_1",
    "oc_spread",
    "hl_spread",
    "vol_chg",
    "sma_5_gap",
    "sma_20_gap",
    "ema_10_gap",
    "vol_10",
    "momentum_5",
    "rsi_14",
]

EXTRA_FEATURES: List[str] = ["volume_regime", "vol_regime", "price_position"]

# Back-compat alias — some callers import the base set under the old name.
FEATURE_COLUMNS: List[str] = BASE_FEATURES


def feature_columns(env_cfg: Dict[str, Any]) -> List[str]:
    """Active feature list = base features plus any enabled, recognised extras."""
    requested = list(env_cfg.get("extra_features", []) or [])
    extras = [name for name in requested if name in EXTRA_FEATURES]
    return [*BASE_FEATURES, *extras]


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _add_extra_features(df: pd.DataFrame, names: Iterable[str]) -> None:
    close = df["close"]
    for name in names:
        if name == "volume_regime":
            df["volume_regime"] = df["volume"] / (df["volume"].rolling(20, min_periods=5).mean() + 1e-12) - 1.0
        elif name == "vol_regime":
            short = df["ret_1"].rolling(5, min_periods=2).std()
            long = df["ret_1"].rolling(20, min_periods=5).std()
            df["vol_regime"] = short / (long + 1e-12)
        elif name == "price_position":
            window = close.rolling(252, min_periods=20)
            lo = window.min()
            hi = window.max()
            df["price_position"] = (close - lo) / ((hi - lo) + 1e-12)


def _add_features(frame: pd.DataFrame, extra_features: Iterable[str] | None = None) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"]

    df["ret_1"] = close.pct_change()
    df["log_ret_1"] = np.log(close / close.shift(1))
    df["oc_spread"] = (df["close"] - df["open"]) / (df["open"] + 1e-12)
    df["hl_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)
    df["vol_chg"] = np.log(df["volume"] + 1.0).diff()
    df["sma_5_gap"] = close / close.rolling(5).mean() - 1.0
    df["sma_20_gap"] = close / close.rolling(20).mean() - 1.0
    df["ema_10_gap"] = close / close.ewm(span=10, adjust=False).mean() - 1.0
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["momentum_5"] = close / close.shift(5) - 1.0
    df["rsi_14"] = _compute_rsi(close, period=14) / 100.0

    enabled = [name for name in (extra_features or []) if name in EXTRA_FEATURES]
    _add_extra_features(df, enabled)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
