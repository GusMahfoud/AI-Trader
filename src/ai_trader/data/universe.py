"""Multi-ticker panel loading for cross-sectional strategies.

Loads each universe ticker through the existing single-ticker loader (and its
Parquet cache) and stacks them into one long panel: one row per (date, ticker).

Survivorship caveat (documented, deliberate): yfinance has no point-in-time
index membership, so the universe is a fixed list of names that existed at the
backtest start AND survived to today. Backtest results carry an optimistic bias
from this and must be read accordingly; the mitigation is a documented, frozen
list rather than a curated-with-hindsight one.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ai_trader.utils.logging import get_logger

from .loader import _load_market_data

logger = get_logger(__name__)

TICKER_COL = "ticker"


def universe_tickers(env_cfg: Dict[str, Any]) -> List[str]:
    """The configured universe, falling back to the single-ticker config."""
    universe = list(env_cfg.get("universe", []) or [])
    if not universe:
        universe = [str(env_cfg.get("ticker", "AAPL"))]
    return universe


def load_universe_panel(config: Dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV for every universe ticker into a long (date, ticker) panel.

    Tickers whose history fails to load or is too short are dropped with a
    warning instead of failing the whole panel. Synthetic sources get a distinct
    seed per ticker so test panels contain genuinely different series.
    """
    env_cfg = dict(config.get("env", {}))
    seed = int(config.get("training", {}).get("seed", 42))
    tickers = universe_tickers(env_cfg)

    frames: List[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        # Adjusted closes are non-negotiable for cross-ticker momentum (splits).
        ticker_cfg = {**env_cfg, "ticker": ticker, "auto_adjust": True}
        try:
            frame = _load_market_data(ticker_cfg, seed=seed + i)
        except (ValueError, ImportError) as exc:
            logger.warning(f"Dropping {ticker} from universe: {exc}")
            continue
        frame = frame.copy()
        frame[TICKER_COL] = ticker
        frames.append(frame)

    if not frames:
        raise ValueError("No universe ticker loaded successfully.")

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["date", TICKER_COL]).reset_index(drop=True)


def close_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long panel to a date x ticker matrix of closes (NaN = not traded)."""
    return panel.pivot(index="date", columns=TICKER_COL, values="close").sort_index()
