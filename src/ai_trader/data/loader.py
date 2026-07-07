"""Market data loading: CSV, yfinance, and synthetic OHLCV sources."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .cache import cache_path, read_cache, write_cache

REQUIRED_OHLCV = ["open", "high", "low", "close", "volume"]


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
        {"date": dates, "open": open_, "high": high, "low": low,
         "close": close, "volume": volume}
    )


def _load_market_data(env_cfg: Dict[str, Any], seed: int) -> pd.DataFrame:
    source = str(env_cfg.get("data_source", "synthetic")).lower()

    # yfinance is the only network source — serve a cleaned cache when available.
    cpath = cache_path(env_cfg) if source == "yfinance" else None
    if cpath is not None and not bool(env_cfg.get("refresh_data", False)):
        cached = read_cache(cpath)
        if cached is not None:
            return cached

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
                "yfinance is required for env.data_source='yfinance'. "
                "Install with: pip install yfinance"
            ) from exc

        ticker = str(env_cfg.get("ticker", "AAPL"))
        start = str(env_cfg.get("start_date", "2015-01-01"))
        end = env_cfg.get("end_date")
        # auto_adjust folds splits/dividends into close — required for any
        # multi-year cross-ticker comparison (raw closes fake huge returns at splits).
        adjust = bool(env_cfg.get("auto_adjust", False))
        frame = yf.download(ticker, start=start, end=end, auto_adjust=adjust, progress=False)
        if frame.empty:
            raise ValueError(
                f"No market data returned for ticker={ticker} start={start} end={end}."
            )
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

    if cpath is not None:
        write_cache(cpath, frame)

    return frame
