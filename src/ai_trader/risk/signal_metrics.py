"""Cross-sectional signal quality metrics: IC, IC IR, NDCG@K, top-K spread.

These evaluate a *ranking* signal (per-date scores over a universe of tickers)
against realized forward returns — the standard yardsticks for cross-sectional
strategies, independent of any portfolio construction.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

DATE_COL = "date"
SCORE_COL = "score"
LABEL_COL = "fwd_return"


def spearman_ic(scores: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank correlation between one date's scores and realized returns."""
    s = pd.Series(scores).rank()
    l = pd.Series(labels).rank()
    if s.nunique() < 2 or l.nunique() < 2:
        return 0.0
    return float(np.corrcoef(s, l)[0, 1])


def ic_series(panel: pd.DataFrame) -> pd.Series:
    """Per-date Spearman IC over a long panel with date/score/fwd_return columns."""
    return panel.groupby(DATE_COL, sort=True)[[SCORE_COL, LABEL_COL]].apply(
        lambda g: spearman_ic(g[SCORE_COL].to_numpy(), g[LABEL_COL].to_numpy())
    )


def ic_summary(ic: pd.Series) -> Dict[str, float]:
    """Aggregate an IC series into mean, std, IR (mean/std), and hit rate."""
    if ic.empty:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_hit_rate": 0.0}
    mean = float(ic.mean())
    std = float(ic.std(ddof=0))
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": mean / std if std > 0 else 0.0,
        "ic_hit_rate": float((ic > 0).mean()),
    }


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """NDCG@K for one date, using the label's cross-sectional percentile as gain.

    1.0 means the top-K by score are exactly the top-K by realized return.
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    # Percentile-of-return relevance keeps gains non-negative and scale-free.
    relevance = pd.Series(labels).rank(pct=True).to_numpy()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    order_by_score = np.argsort(-scores)[:k]
    dcg = float(np.sum(relevance[order_by_score] * discounts))
    ideal = float(np.sum(np.sort(relevance)[::-1][:k] * discounts))
    return dcg / ideal if ideal > 0 else 0.0


def ndcg_series(panel: pd.DataFrame, k: int) -> pd.Series:
    """Per-date NDCG@K over a long panel."""
    return panel.groupby(DATE_COL, sort=True)[[SCORE_COL, LABEL_COL]].apply(
        lambda g: ndcg_at_k(g[SCORE_COL].to_numpy(), g[LABEL_COL].to_numpy(), k)
    )


def topk_spread_series(panel: pd.DataFrame, k: int) -> pd.Series:
    """Per-date mean fwd_return of the top-K by score minus the bottom-K by score."""

    def spread(g: pd.DataFrame) -> float:
        if len(g) < 2 * k:
            return 0.0
        ordered = g.sort_values(SCORE_COL, ascending=False)
        top = float(ordered[LABEL_COL].head(k).mean())
        bottom = float(ordered[LABEL_COL].tail(k).mean())
        return top - bottom

    return panel.groupby(DATE_COL, sort=True)[[SCORE_COL, LABEL_COL]].apply(spread)
