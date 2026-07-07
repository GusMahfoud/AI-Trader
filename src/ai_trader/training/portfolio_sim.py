"""Daily portfolio simulator for top-K ranked portfolios.

Pure equity-curve arithmetic: no data loading, no model. Trades happen at the
close of each rebalance date using scores computed from that date's (backward-
looking) features; returns accrue from the next close onward, so there is no
same-bar look-ahead.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def _target_weights(
    tickers: pd.Index, scores: Optional[pd.Series], top_k: Optional[int]
) -> pd.Series:
    """Equal weights over the top-K by score (or over all names when top_k is None)."""
    if top_k is None or scores is None:
        held = list(tickers)
    else:
        held = list(scores.dropna().nlargest(top_k).index)
    weights = pd.Series(0.0, index=tickers)
    if held:
        weights[held] = 1.0 / len(held)
    return weights


def simulate_rank_portfolio(
    closes: pd.DataFrame,
    score_panel: Optional[pd.DataFrame],
    top_k: Optional[int],
    rebalance_days: int,
    cost_rate: float,
) -> np.ndarray:
    """Simulate a (re)ranked portfolio over `closes` (date x ticker, ascending).

    ``score_panel`` is a long frame with date/ticker/score columns; None means
    an equal-weight-everything benchmark. Costs charge ``cost_rate`` per unit of
    traded weight (buys and sells both pay). Returns the equity curve (start 1.0).
    """
    dates = closes.index
    if len(dates) < 2:
        return np.ones(len(dates))

    scores_by_date = None
    if score_panel is not None:
        scores_by_date = {
            d: g.set_index("ticker")["score"] for d, g in score_panel.groupby("date")
        }

    weights = pd.Series(0.0, index=closes.columns)
    daily_rets = closes.pct_change().fillna(0.0)
    equity: List[float] = [1.0]

    for i, date in enumerate(dates[:-1]):
        if i % rebalance_days == 0:
            day_scores = scores_by_date.get(date) if scores_by_date is not None else None
            if scores_by_date is None or day_scores is not None:
                target = _target_weights(closes.columns, day_scores, top_k)
                traded = float((target - weights).abs().sum())
                weights = target
                equity[-1] = equity[-1] * (1.0 - traded * cost_rate)

        rets = daily_rets.iloc[i + 1]
        port_ret = float((weights * rets).sum())
        equity.append(equity[-1] * (1.0 + port_ret))

        # Drift: winners grow their weight until the next rebalance.
        if port_ret > -1.0:
            weights = weights * (1.0 + rets) / (1.0 + port_ret)

    return np.asarray(equity)
