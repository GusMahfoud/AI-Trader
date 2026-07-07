"""Tests for the top-K portfolio simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trader.training.portfolio_sim import simulate_rank_portfolio


@pytest.fixture
def closes():
    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    up = 100.0 * (1.01 ** np.arange(60))
    down = 100.0 * (0.99 ** np.arange(60))
    return pd.DataFrame({"UP": up, "DOWN": down}, index=dates)


def _scores(closes: pd.DataFrame, favored: str) -> pd.DataFrame:
    rows = []
    for d in closes.index:
        for t in closes.columns:
            rows.append({"date": d, "ticker": t, "score": 1.0 if t == favored else 0.0})
    return pd.DataFrame(rows)


def test_picks_winner_beats_equal_weight(closes):
    strat = simulate_rank_portfolio(closes, _scores(closes, "UP"), top_k=1, rebalance_days=20, cost_rate=0.0)
    ew = simulate_rank_portfolio(closes, None, top_k=None, rebalance_days=20, cost_rate=0.0)
    assert strat[-1] > ew[-1] > 0.0


def test_picks_loser_loses(closes):
    strat = simulate_rank_portfolio(closes, _scores(closes, "DOWN"), top_k=1, rebalance_days=20, cost_rate=0.0)
    assert strat[-1] < 1.0


def test_costs_reduce_equity(closes):
    free = simulate_rank_portfolio(closes, _scores(closes, "UP"), top_k=1, rebalance_days=5, cost_rate=0.0)
    costly = simulate_rank_portfolio(closes, _scores(closes, "UP"), top_k=1, rebalance_days=5, cost_rate=0.01)
    assert costly[-1] < free[-1]


def test_no_same_bar_lookahead(closes):
    """Day-one equity must be unaffected by day-one returns (trade at close)."""
    equity = simulate_rank_portfolio(closes, _scores(closes, "UP"), top_k=1, rebalance_days=20, cost_rate=0.0)
    assert equity[0] == pytest.approx(1.0)
    assert len(equity) == len(closes)


def test_single_date_is_flat():
    dates = pd.date_range("2021-01-01", periods=1, freq="B")
    closes = pd.DataFrame({"A": [100.0]}, index=dates)
    equity = simulate_rank_portfolio(closes, None, top_k=None, rebalance_days=5, cost_rate=0.001)
    assert list(equity) == [1.0]
