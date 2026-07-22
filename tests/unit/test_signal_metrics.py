"""Tests for cross-sectional signal metrics (IC, NDCG@K, top-K spread)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trader.risk.signal_metrics import (
    ic_series,
    ic_summary,
    ndcg_at_k,
    spearman_ic,
    topk_spread_series,
)


def _panel(n_dates: int = 5, n_tickers: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2020-01-01", periods=n_dates, freq="B"):
        labels = rng.normal(0, 0.02, n_tickers)
        for i in range(n_tickers):
            rows.append({"date": d, "ticker": f"T{i}", "score": 0.0, "fwd_return": labels[i]})
    return pd.DataFrame(rows)


def test_spearman_perfect_and_inverted():
    labels = np.array([0.01, 0.03, -0.02, 0.05, 0.0])
    assert spearman_ic(labels.copy(), labels) == pytest.approx(1.0)
    assert spearman_ic(-labels, labels) == pytest.approx(-1.0)


def test_spearman_degenerate_scores():
    labels = np.array([0.01, 0.02, 0.03])
    assert spearman_ic(np.zeros(3), labels) == 0.0


def test_ic_series_perfect_signal():
    panel = _panel()
    panel["score"] = panel["fwd_return"]
    ic = ic_series(panel)
    assert len(ic) == 5
    assert np.allclose(ic.to_numpy(), 1.0)
    summary = ic_summary(ic)
    assert summary["ic_mean"] == pytest.approx(1.0)
    assert summary["ic_hit_rate"] == pytest.approx(1.0)


def test_ic_summary_empty():
    summary = ic_summary(pd.Series(dtype=float))
    assert summary == {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_hit_rate": 0.0}


def test_ndcg_perfect_ranking_is_one():
    labels = np.array([0.05, 0.03, 0.01, -0.01, -0.03])
    assert ndcg_at_k(labels.copy(), labels, k=2) == pytest.approx(1.0)


def test_ndcg_worst_ranking_below_one():
    labels = np.array([0.05, 0.03, 0.01, -0.01, -0.03])
    worst = ndcg_at_k(-labels, labels, k=2)
    assert 0.0 <= worst < 1.0


def test_ndcg_empty():
    assert ndcg_at_k(np.array([]), np.array([]), k=3) == 0.0


def test_topk_spread_positive_for_aligned_scores():
    panel = _panel(n_tickers=10)
    panel["score"] = panel["fwd_return"]
    spread = topk_spread_series(panel, k=3)
    assert (spread > 0).all()


def test_topk_spread_too_few_names_is_zero():
    panel = _panel(n_tickers=4)
    panel["score"] = panel["fwd_return"]
    spread = topk_spread_series(panel, k=3)
    assert (spread == 0.0).all()
