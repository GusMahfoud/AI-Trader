"""Tests for the LambdaRank scorer: learnable signal in, ranking skill out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trader.data.cross_features import RANK_COLUMNS
from ai_trader.data.labels import LABEL_BIN_COL
from ai_trader.models.ranker import LambdaRankScorer
from ai_trader.risk.signal_metrics import spearman_ic


def _labeled_panel(n_dates: int, n_tickers: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic panel where one feature drives the label and the rest are noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2019-01-01", periods=n_dates, freq="B"):
        signal = rng.uniform(0, 1, n_tickers)
        noise = {c: rng.uniform(0, 1, n_tickers) for c in RANK_COLUMNS if c != "rank_mom_12_1"}
        bins = pd.Series(signal).rank(pct=True).apply(lambda v: min(int(v * 4), 3))
        for i in range(n_tickers):
            row = {"date": d, "ticker": f"T{i}", "rank_mom_12_1": signal[i], LABEL_BIN_COL: bins[i]}
            for c, vals in noise.items():
                row[c] = vals[i]
            rows.append(row)
    return pd.DataFrame(rows)


def test_ranker_learns_planted_signal():
    train = _labeled_panel(n_dates=100, seed=1)
    test = _labeled_panel(n_dates=20, seed=2)

    scorer = LambdaRankScorer(seed=42, params={"min_child_samples": 20}).fit(train)
    scores = scorer.score(test)

    # Scores must recover the planted feature's ordering out of sample.
    ic = spearman_ic(scores, test["rank_mom_12_1"].to_numpy())
    assert ic > 0.5


def test_ranker_requires_fit_before_score():
    with pytest.raises(RuntimeError):
        LambdaRankScorer(seed=42).score(_labeled_panel(n_dates=5))


def test_ranker_rejects_unlabeled_panel():
    panel = _labeled_panel(n_dates=5)
    panel[LABEL_BIN_COL] = np.nan
    with pytest.raises(ValueError):
        LambdaRankScorer(seed=42).fit(panel)
