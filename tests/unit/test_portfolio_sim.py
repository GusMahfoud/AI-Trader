"""Tests for the top-K portfolio simulator and its Phase-5 rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trader.training.portfolio_sim import PortfolioRules, simulate_rank_portfolio


@pytest.fixture
def closes():
    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    up = 100.0 * (1.01 ** np.arange(60))
    down = 100.0 * (0.99 ** np.arange(60))
    return pd.DataFrame({"UP": up, "DOWN": down}, index=dates)


def _scores(closes: pd.DataFrame, favored: str, vol: dict | None = None) -> pd.DataFrame:
    rows = []
    for d in closes.index:
        for t in closes.columns:
            row = {"date": d, "ticker": t, "score": 1.0 if t == favored else 0.0}
            if vol is not None:
                row["vol"] = vol[t]
            rows.append(row)
    return pd.DataFrame(rows)


def _rules(**kwargs) -> PortfolioRules:
    defaults = dict(top_k=1, rebalance_days=20, cost_rate=0.0)
    return PortfolioRules(**{**defaults, **kwargs})


def test_picks_winner_beats_equal_weight(closes):
    strat = simulate_rank_portfolio(closes, _scores(closes, "UP"), _rules())
    ew = simulate_rank_portfolio(closes, None, _rules(top_k=None))
    assert strat[-1] > ew[-1] > 0.0


def test_picks_loser_loses(closes):
    strat = simulate_rank_portfolio(closes, _scores(closes, "DOWN"), _rules())
    assert strat[-1] < 1.0


def test_costs_reduce_equity(closes):
    free = simulate_rank_portfolio(closes, _scores(closes, "UP"), _rules(rebalance_days=5))
    costly = simulate_rank_portfolio(
        closes, _scores(closes, "UP"), _rules(rebalance_days=5, cost_rate=0.01)
    )
    assert costly[-1] < free[-1]


def test_no_same_bar_lookahead(closes):
    """Day-one equity must be unaffected by day-one returns (trade at close)."""
    equity = simulate_rank_portfolio(closes, _scores(closes, "UP"), _rules())
    assert equity[0] == pytest.approx(1.0)
    assert len(equity) == len(closes)


def test_single_date_is_flat():
    dates = pd.date_range("2021-01-01", periods=1, freq="B")
    closes = pd.DataFrame({"A": [100.0]}, index=dates)
    equity = simulate_rank_portfolio(closes, None, _rules(top_k=None, rebalance_days=5))
    assert list(equity) == [1.0]


def test_buffer_keeps_incumbent(closes):
    """With a buffer, a rank-2 incumbent is kept; without it, it is swapped."""
    dates = closes.index
    rows = []
    for i, d in enumerate(dates):
        # DOWN starts as the top pick, then drops to rank 2 forever.
        down_score = 2.0 if i < 5 else 0.5
        rows.append({"date": d, "ticker": "DOWN", "score": down_score})
        rows.append({"date": d, "ticker": "UP", "score": 1.0})
    panel = pd.DataFrame(rows)

    no_buffer = simulate_rank_portfolio(closes, panel, _rules(rebalance_days=5))
    buffered = simulate_rank_portfolio(closes, panel, _rules(rebalance_days=5, buffer_k=2))
    # Buffered portfolio never sells DOWN (still within top-2), so it keeps losing;
    # unbuffered swaps to UP and ends higher.
    assert no_buffer[-1] > buffered[-1]


def test_inverse_vol_tilts_to_calm_name(closes):
    scores = _scores(closes, "UP", vol={"UP": 0.01, "DOWN": 0.04})
    scores["score"] = 1.0  # tie: both held with top_k=2
    eq = simulate_rank_portfolio(closes, scores, _rules(top_k=2))
    iv = simulate_rank_portfolio(
        closes, scores, _rules(top_k=2, weighting="inverse_vol")
    )
    # UP is the calm name here, so inverse-vol overweights the winner.
    assert iv[-1] > eq[-1]


def test_dd_brake_cuts_exposure_in_decline(closes):
    braked = simulate_rank_portfolio(
        closes, _scores(closes, "DOWN"), _rules(rebalance_days=5, dd_brake=0.05)
    )
    unbraked = simulate_rank_portfolio(
        closes, _scores(closes, "DOWN"), _rules(rebalance_days=5)
    )
    assert braked[-1] > unbraked[-1]
