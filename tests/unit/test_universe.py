"""Tests for the multi-ticker universe panel loader."""

from __future__ import annotations

import pytest

from ai_trader.data.universe import close_matrix, load_universe_panel, universe_tickers


@pytest.fixture
def universe_config(synthetic_config):
    cfg = {**synthetic_config}
    cfg["env"] = {**synthetic_config["env"], "universe": ["AAA", "BBB", "CCC"]}
    return cfg


def test_universe_tickers_fallback(synthetic_config):
    assert universe_tickers({**synthetic_config["env"], "ticker": "AAPL"}) == ["AAPL"]


def test_panel_has_all_tickers(universe_config):
    panel = load_universe_panel(universe_config)
    assert sorted(panel["ticker"].unique()) == ["AAA", "BBB", "CCC"]
    assert {"date", "open", "high", "low", "close", "volume", "ticker"} <= set(panel.columns)


def test_panel_series_differ_per_ticker(universe_config):
    panel = load_universe_panel(universe_config)
    closes = close_matrix(panel)
    # Per-ticker seeds must produce distinct synthetic series.
    assert not closes["AAA"].equals(closes["BBB"])


def test_close_matrix_shape(universe_config):
    panel = load_universe_panel(universe_config)
    closes = close_matrix(panel)
    assert closes.shape[1] == 3
    assert closes.index.is_monotonic_increasing
