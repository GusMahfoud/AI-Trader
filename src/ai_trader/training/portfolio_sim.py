"""Daily portfolio simulator for top-K ranked portfolios.

Pure equity-curve arithmetic: no data loading, no model. Trades happen at the
close of each rebalance date using scores computed from that date's (backward-
looking) features; returns accrue from the next close onward, so there is no
same-bar look-ahead.

Phase-5 rules (all optional, pre-committed defaults in config.yaml):
- Turnover buffer: an incumbent holding is kept while it stays inside the top
  ``buffer_k`` ranks, even if it leaves the top K — cuts churn and cost drag.
- Inverse-vol weighting: capital per name proportional to 1/volatility.
- Drawdown brake: when the portfolio sits more than ``dd_brake`` below its
  peak, target weights are halved at the next rebalance until recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BRAKE_SCALE = 0.5


@dataclass(frozen=True)
class PortfolioRules:
    """Rebalance behaviour for `simulate_rank_portfolio` (defaults = Phase-4 behaviour)."""

    top_k: Optional[int] = 5           # None = hold every available name
    rebalance_days: int = 21
    cost_rate: float = 0.0004          # per unit of traded weight
    buffer_k: int = 0                  # 0 = off; else keep incumbents while rank <= buffer_k
    weighting: str = "equal"           # "equal" | "inverse_vol"
    dd_brake: float = 0.0              # 0 = off; else drawdown fraction that halves exposure


def _select_holdings(
    scores: pd.Series, held: List[str], rules: PortfolioRules
) -> List[str]:
    """Top-K selection with an optional incumbency buffer."""
    ranked = list(scores.dropna().sort_values(ascending=False).index)
    if rules.top_k is None:
        return ranked
    if rules.buffer_k > rules.top_k:
        buffer_zone = set(ranked[: rules.buffer_k])
        keep = [t for t in held if t in buffer_zone]
        fresh = [t for t in ranked if t not in keep]
        return (keep + fresh)[: rules.top_k]
    return ranked[: rules.top_k]


def _target_weights(
    tickers: pd.Index,
    held_names: List[str],
    vols: Optional[pd.Series],
    rules: PortfolioRules,
    scale: float,
) -> pd.Series:
    """Weights over the chosen names: equal or inverse-volatility, times brake scale."""
    weights = pd.Series(0.0, index=tickers)
    if not held_names:
        return weights
    if rules.weighting == "inverse_vol" and vols is not None:
        inv = (1.0 / vols.reindex(held_names)).replace([np.inf, -np.inf], np.nan)
        inv = inv.fillna(inv.mean() if inv.notna().any() else 1.0)
        weights[held_names] = (inv / inv.sum()).to_numpy()
    else:
        weights[held_names] = 1.0 / len(held_names)
    return weights * scale


def simulate_rank_portfolio(
    closes: pd.DataFrame,
    score_panel: Optional[pd.DataFrame],
    rules: PortfolioRules,
) -> np.ndarray:
    """Simulate a (re)ranked portfolio over `closes` (date x ticker, ascending).

    ``score_panel`` is a long frame with date/ticker/score (and optionally vol)
    columns; None means an equal-weight-everything benchmark. Costs charge
    ``rules.cost_rate`` per unit of traded weight (buys and sells both pay).
    Returns the equity curve (start 1.0).
    """
    dates = closes.index
    if len(dates) < 2:
        return np.ones(len(dates))

    by_date: Optional[Dict] = None
    if score_panel is not None:
        by_date = {d: g.set_index("ticker") for d, g in score_panel.groupby("date")}

    weights = pd.Series(0.0, index=closes.columns)
    daily_rets = closes.pct_change(fill_method=None).fillna(0.0)
    equity: List[float] = [1.0]
    peak = 1.0

    for i, date in enumerate(dates[:-1]):
        if i % rules.rebalance_days == 0:
            day = by_date.get(date) if by_date is not None else None
            if by_date is None or day is not None:
                if by_date is None:
                    names = list(closes.columns[closes.loc[date].notna()])
                    vols = None
                else:
                    held = list(weights.index[weights > 0])
                    names = _select_holdings(day["score"], held, rules)
                    vols = day["vol"] if "vol" in day.columns else None

                braked = rules.dd_brake > 0 and (peak - equity[-1]) / peak > rules.dd_brake
                scale = BRAKE_SCALE if braked else 1.0
                target = _target_weights(closes.columns, names, vols, rules, scale)
                traded = float((target - weights).abs().sum())
                weights = target
                equity[-1] = equity[-1] * (1.0 - traded * rules.cost_rate)

        rets = daily_rets.iloc[i + 1]
        port_ret = float((weights * rets).sum())
        equity.append(equity[-1] * (1.0 + port_ret))
        peak = max(peak, equity[-1])

        if port_ret > -1.0:
            weights = weights * (1.0 + rets) / (1.0 + port_ret)

    return np.asarray(equity)
