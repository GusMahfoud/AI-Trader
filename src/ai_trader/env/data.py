"""Market-data loading, feature engineering, and chronological splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_OHLCV = ["open", "high", "low", "close", "volume"]

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


@dataclass
class DataBundle:
    frame: pd.DataFrame
    feature_columns: List[str]
    split_bounds: Dict[str, Tuple[int, int]]
    feature_mean: pd.Series
    feature_std: pd.Series


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _generate_synthetic_data(length: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = 0.0003
    vol = 0.015
    returns = rng.normal(loc=drift, scale=vol, size=length)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=length))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.006, size=length))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.006, size=length))
    volume = rng.integers(1_000_000, 4_000_000, size=length)
    dates = pd.date_range("2012-01-01", periods=length, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _load_market_data(env_cfg: Dict[str, Any], seed: int) -> pd.DataFrame:
    source = str(env_cfg.get("data_source", "synthetic")).lower()

    if source == "csv":
        data_path = env_cfg.get("data_path")
        if not data_path:
            raise ValueError("env.data_path is required when env.data_source='csv'.")
        frame = pd.read_csv(data_path)
    elif source == "yfinance":
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for env.data_source='yfinance'. Install with: pip install yfinance"
            ) from exc

        ticker = str(env_cfg.get("ticker", "AAPL"))
        start = str(env_cfg.get("start_date", "2015-01-01"))
        end = env_cfg.get("end_date")
        frame = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if frame.empty:
            raise ValueError(f"No market data returned for ticker={ticker} start={start} end={end}.")
        frame = frame.reset_index().rename(columns={"Date": "date"})
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(c[0]).lower() for c in frame.columns]
        else:
            frame.columns = [str(c).lower() for c in frame.columns]
    elif source == "synthetic":
        length = int(env_cfg.get("synthetic_length", 3000))
        frame = _generate_synthetic_data(length=length, seed=seed)
    else:
        raise ValueError("env.data_source must be one of: 'csv', 'yfinance', 'synthetic'.")

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "date" not in frame.columns:
        frame["date"] = pd.date_range("2000-01-01", periods=len(frame), freq="B")

    missing = [c for c in REQUIRED_OHLCV if c not in frame.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    frame = frame[["date", *REQUIRED_OHLCV]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for col in REQUIRED_OHLCV:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna().sort_values("date").reset_index(drop=True)

    if len(frame) < 200:
        raise ValueError("Market data too short. Provide at least 200 rows.")

    return frame


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

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def _split_bounds(length: int, train_ratio: float, val_ratio: float) -> Dict[str, Tuple[int, int]]:
    if not (0.4 < train_ratio < 0.95):
        raise ValueError("env.train_ratio must be in (0.4, 0.95).")
    if not (0.0 < val_ratio < 0.4):
        raise ValueError("env.val_ratio must be in (0.0, 0.4).")
    if train_ratio + val_ratio >= 0.98:
        raise ValueError("env.train_ratio + env.val_ratio must be < 0.98.")

    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }


def build_data_bundle(config: Dict[str, Any]) -> DataBundle:
    env_cfg = config.get("env", {})
    seed = int(config.get("training", {}).get("seed", 42))

    raw = _load_market_data(env_cfg, seed=seed)
    frame = _add_features(raw)

    splits = _split_bounds(
        length=len(frame),
        train_ratio=float(env_cfg.get("train_ratio", 0.7)),
        val_ratio=float(env_cfg.get("val_ratio", 0.15)),
    )

    # Normalize using training-split statistics only — prevents look-ahead leakage.
    t0, t1 = splits["train"]
    train_slice = frame.iloc[t0:t1]
    mu = train_slice[FEATURE_COLUMNS].mean()
    sigma = train_slice[FEATURE_COLUMNS].std().replace(0.0, 1.0).fillna(1.0)

    frame = frame.copy()
    frame[FEATURE_COLUMNS] = (frame[FEATURE_COLUMNS] - mu) / sigma

    return DataBundle(
        frame=frame,
        feature_columns=FEATURE_COLUMNS,
        split_bounds=splits,
        feature_mean=mu,
        feature_std=sigma,
    )
