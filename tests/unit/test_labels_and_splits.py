"""Tests for forward-return labels and purged walk-forward splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trader.data.labels import LABEL_BIN_COL, LABEL_COL, add_forward_returns, add_label_bins
from ai_trader.data.purged_splits import assert_no_label_overlap, purged_walk_forward_bounds


@pytest.fixture
def panel():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rows = []
    for ticker, step in [("AAA", 1.0), ("BBB", -0.5)]:
        for i, d in enumerate(dates):
            rows.append(
                {"date": d, "ticker": ticker, "open": 0, "high": 0, "low": 0,
                 "close": 100.0 + step * i, "volume": 1}
            )
    return pd.DataFrame(rows)


def test_forward_return_values(panel):
    out = add_forward_returns(panel, horizon=10)
    aaa = out[out["ticker"] == "AAA"].sort_values("date")
    # close goes 100, 101, ... so fwd 10d return at t=0 is 110/100 - 1.
    assert aaa[LABEL_COL].iloc[0] == pytest.approx(0.10)
    # Last `horizon` rows have no future close: label must be NaN, never filled.
    assert aaa[LABEL_COL].tail(10).isna().all()


def test_label_bins_are_per_date_ordinals(panel):
    out = add_label_bins(add_forward_returns(panel, horizon=5), n_bins=2)
    one_date = out[out["date"] == out["date"].iloc[0]]
    # AAA rises (better fwd return) -> top bin; BBB falls -> bottom bin.
    assert one_date.set_index("ticker")[LABEL_BIN_COL]["AAA"] == 1
    assert one_date.set_index("ticker")[LABEL_BIN_COL]["BBB"] == 0


def test_purged_bounds_tile_and_purge():
    folds = purged_walk_forward_bounds(n_dates=1000, n_splits=4, label_horizon=21)
    assert len(folds) == 4
    # Test blocks tile the tail contiguously and end at the last date.
    for a, b in zip(folds, folds[1:]):
        assert a["test"][1] == b["test"][0]
    assert folds[-1]["test"][1] == 1000
    # Purge gap: training always ends label_horizon before the test block.
    for f in folds:
        assert f["train"][1] == f["test"][0] - 21
    assert_no_label_overlap(folds, label_horizon=21)


def test_purged_bounds_leakage_assertion_fires():
    bad = [{"train": (0, 500), "test": (510, 600)}]
    with pytest.raises(AssertionError):
        assert_no_label_overlap(bad, label_horizon=21)


def test_purged_bounds_insufficient_data():
    with pytest.raises(ValueError):
        purged_walk_forward_bounds(n_dates=120, n_splits=8, label_horizon=21)
