"""Forward-return labels for cross-sectional ranking.

A row's label describes the FUTURE window [t, t+horizon] — any split that puts
a labeled row in training while that window overlaps a test period is leakage.
`purged_splits.py` enforces the matching purge gap.
"""

from __future__ import annotations

import pandas as pd

from .universe import TICKER_COL

LABEL_COL = "fwd_return"
LABEL_BIN_COL = "fwd_return_bin"


def add_forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Append each stock's forward ``horizon``-day simple return (NaN near the end)."""
    out = panel.sort_values([TICKER_COL, "date"]).copy()
    future_close = out.groupby(TICKER_COL, sort=False)["close"].shift(-horizon)
    out[LABEL_COL] = future_close / out["close"] - 1.0
    return out.sort_values(["date", TICKER_COL]).reset_index(drop=True)


def add_label_bins(panel: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Append per-date quantile bins of the forward return (0 = worst, n-1 = best).

    Rankers (e.g. LightGBM lambdarank) want small ordinal relevance grades, not
    raw returns; binning within each date also removes market-direction effects.
    """
    out = panel.copy()

    def bin_date(s: pd.Series) -> pd.Series:
        ranked = s.rank(pct=True)
        return (ranked * n_bins).clip(upper=n_bins).apply(
            lambda v: int(v - 1e-9) if pd.notna(v) else v
        )

    out[LABEL_BIN_COL] = out.groupby("date", sort=False)[LABEL_COL].transform(bin_date)
    return out
