"""Feature engineering: technical indicators computed from OHLCV data."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
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


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
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

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
