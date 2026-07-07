"""Tests for cross-sectional features: correctness and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trader.data.cross_features import (
    CROSS_FEATURES,
    RANK_COLUMNS,
    add_stock_features,
    build_cross_features,
)


@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-01", periods=300, freq="B")
    rows = []
    for ticker, drift in [("AAA", 0.001), ("BBB", 0.0), ("CCC", -0.001)]:
        close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(dates))))
        for d, c in zip(dates, close):
            rows.append(
                {"date": d, "ticker": ticker, "open": c, "high": c, "low": c,
                 "close": c, "volume": 1_000_000}
            )
    return pd.DataFrame(rows)


def test_features_present_and_complete(panel):
    out = build_cross_features(panel)
    for col in CROSS_FEATURES + RANK_COLUMNS:
        assert col in out.columns
        assert out[col].notna().all()


def test_warmup_rows_dropped(panel):
    out = build_cross_features(panel)
    # 12-1 momentum needs 252 prior days; nothing before that can survive.
    assert out["date"].min() > panel["date"].min() + pd.Timedelta(days=300)


def test_momentum_no_lookahead(panel):
    """Truncating the future must not change a past date's feature values."""
    full = add_stock_features(panel)
    truncated = add_stock_features(panel[panel["date"] <= "2019-01-01"])
    cutoff = pd.Timestamp("2018-12-01")
    a = full[(full["date"] <= cutoff) & (full["ticker"] == "AAA")]["mom_21"].reset_index(drop=True)
    b = truncated[(truncated["date"] <= cutoff) & (truncated["ticker"] == "AAA")]["mom_21"].reset_index(drop=True)
    pd.testing.assert_series_equal(a, b)


def test_ranks_are_per_date_percentiles(panel):
    out = build_cross_features(panel)
    one_date = out[out["date"] == out["date"].iloc[-1]]
    ranks = sorted(one_date["rank_mom_21"].tolist())
    assert ranks == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_features_computed_per_ticker(panel):
    """A ticker's momentum must not bleed into another ticker's rows."""
    out = add_stock_features(panel)
    first_bbb = out[out["ticker"] == "BBB"].sort_values("date").iloc[0]
    assert pd.isna(first_bbb["mom_21"])  # BBB's own warmup, not AAA's tail
