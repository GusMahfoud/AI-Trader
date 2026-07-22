"""Cross-sectional features: per-stock signals converted to per-date percentiles.

Raw features are computed per ticker from its own history (backward-looking
only), then each feature is ranked across the universe *within each date* to a
0-1 percentile. Percentiles make stocks directly comparable and need no
train-split scaler, eliminating a whole class of normalization leakage.

Feature sets are versioned so past runs stay reproducible:
- "v1" — the original 7 signals (momentum family, reversal, vol, liquidity).
- "v2" — v1 plus vol-adjusted momentum, 52-week-high distance, up-day ratio,
  and sector-relative ranks of the momentum/reversal signals.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .sectors import sector_of
from .universe import TICKER_COL

# Raw per-stock feature columns (before cross-sectional ranking).
CROSS_FEATURES: List[str] = [
    "mom_21",        # 1-month momentum
    "mom_63",        # 3-month momentum
    "mom_12_1",      # 12-month momentum skipping the most recent month (12-1 convention)
    "ret_5",         # 5-day return (short-term reversal signal)
    "vol_20",        # 20-day realized volatility
    "vol_60",        # 60-day realized volatility
    "dollar_vol_20", # 20-day average dollar volume (liquidity)
]

V2_EXTRA_FEATURES: List[str] = [
    "vol_adj_mom",   # 12-1 momentum per unit of 60d vol (momentum quality)
    "dist_52w_high", # distance below the 252d high (52-week-high anomaly)
    "up_ratio_63",   # fraction of up days over 63d (momentum smoothness)
]

# Signals whose v2 variant is also ranked within sector (macro-bet neutralization).
SECTOR_RANKED_FEATURES: List[str] = ["mom_12_1", "ret_5"]

MAX_FEATURE_LOOKBACK = 252

FEATURE_SETS: Dict[str, List[str]] = {
    "v1": CROSS_FEATURES,
    "v2": CROSS_FEATURES + V2_EXTRA_FEATURES,
}


def rank_col(feature: str) -> str:
    """Name of the cross-sectional percentile column for a raw feature."""
    return f"rank_{feature}"


def sector_rank_col(feature: str) -> str:
    """Name of the within-sector percentile column for a raw feature."""
    return f"srank_{feature}"


RANK_COLUMNS: List[str] = [rank_col(f) for f in CROSS_FEATURES]


def model_feature_columns(feature_set: str = "v1") -> List[str]:
    """Percentile columns a ranking model consumes for the given feature set."""
    features = FEATURE_SETS[feature_set]
    columns = [rank_col(f) for f in features]
    if feature_set == "v2":
        columns += [sector_rank_col(f) for f in SECTOR_RANKED_FEATURES]
    return columns


def add_stock_features(panel: pd.DataFrame, feature_set: str = "v1") -> pd.DataFrame:
    """Append raw per-stock features, computed independently per ticker."""
    out = panel.sort_values([TICKER_COL, "date"]).copy()
    g = out.groupby(TICKER_COL, sort=False)
    close = g["close"]

    out["mom_21"] = close.pct_change(21)
    out["mom_63"] = close.pct_change(63)
    # 12-1: return from t-252 to t-21, skipping the last month (reversal zone).
    out["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    out["ret_5"] = close.pct_change(5)

    daily_ret = close.pct_change()
    by_ticker = lambda s: s.groupby(out[TICKER_COL], sort=False)
    out["vol_20"] = by_ticker(daily_ret).rolling(20).std().reset_index(level=0, drop=True)
    out["vol_60"] = by_ticker(daily_ret).rolling(60).std().reset_index(level=0, drop=True)

    dollar = out["close"] * out["volume"]
    out["dollar_vol_20"] = by_ticker(dollar).rolling(20).mean().reset_index(level=0, drop=True)

    if feature_set == "v2":
        out["vol_adj_mom"] = out["mom_12_1"] / (out["vol_60"] + 1e-8)
        high_252 = by_ticker(out["close"]).rolling(252).max().reset_index(level=0, drop=True)
        out["dist_52w_high"] = out["close"] / high_252 - 1.0
        up_days = (daily_ret > 0).astype(float)
        out["up_ratio_63"] = by_ticker(up_days).rolling(63).mean().reset_index(level=0, drop=True)

    return out.sort_values(["date", TICKER_COL]).reset_index(drop=True)


def add_cross_sectional_ranks(panel: pd.DataFrame, feature_set: str = "v1") -> pd.DataFrame:
    """Append per-date percentile ranks (0-1); v2 also ranks within sectors."""
    out = panel.copy()
    for feature in FEATURE_SETS[feature_set]:
        out[rank_col(feature)] = out.groupby("date", sort=False)[feature].rank(pct=True)

    if feature_set == "v2":
        out["sector"] = out[TICKER_COL].map(sector_of)
        for feature in SECTOR_RANKED_FEATURES:
            out[sector_rank_col(feature)] = (
                out.groupby(["date", "sector"], sort=False)[feature].rank(pct=True)
            )
    return out


def build_cross_features(
    panel: pd.DataFrame, feature_set: str = "v1", dropna: bool = True
) -> pd.DataFrame:
    """Full pipeline: raw per-stock features -> per-date percentiles.

    With ``dropna`` (default) rows inside the max lookback warmup are removed,
    so every remaining row has a complete feature vector.
    """
    out = add_cross_sectional_ranks(add_stock_features(panel, feature_set), feature_set)
    if dropna:
        out = out.dropna(subset=FEATURE_SETS[feature_set]).reset_index(drop=True)
    return out
