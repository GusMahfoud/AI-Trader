"""Live inference: train on all available history, emit today's top-K picks.

This is the paper-trading entry point (`--mode picks`): no folds, no backtest —
it fits the configured scorer on every labeled row up to the present, scores
the most recent trading date, and writes the ranked portfolio (with weights)
to picks.csv. Deliberately CLI-only: it is an interactive read of the model,
not an experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ai_trader.data.cross_features import build_cross_features, model_feature_columns, rank_col
from ai_trader.data.labels import add_forward_returns, add_label_bins
from ai_trader.data.sectors import name_of
from ai_trader.data.universe import load_universe_panel
from ai_trader.models.ranker import LambdaRankScorer
from ai_trader.utils import get_logger

from .portfolio_sim import PortfolioRules, _target_weights
from .rank_backtest import _rank_cfg

logger = get_logger(__name__)

PICKS_FILENAME = "picks.csv"
PAPER_LOG_PATH = Path("paper_trading") / "picks_log.csv"


def _append_paper_log(out: pd.DataFrame, log_path: Path) -> None:
    """Append this month's picks to the running paper-trading log (idempotent
    per as_of date — re-running on the same day replaces that day's rows)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log = pd.read_csv(log_path)
        log = log[log["as_of"] != out["as_of"].iloc[0]]
        log = pd.concat([log, out], ignore_index=True)
    else:
        log = out
    log.to_csv(log_path, index=False)


def _whole_share_allocation(
    weights: np.ndarray, prices: np.ndarray, capital: float
) -> np.ndarray:
    """Integer shares per pick, total cost <= capital, as close to it as possible.

    Floor each pick's weighted target to whole shares, then repeatedly sweep the
    picks in rank order adding one share wherever the remaining cash covers the
    price, until nothing fits. Never exceeds capital; leftover is smaller than
    the cheapest pick's price.
    """
    shares = np.floor(weights * capital / prices).astype(int)
    leftover = capital - float((shares * prices).sum())
    while True:
        added = False
        for i in range(len(prices)):
            if prices[i] <= leftover:
                shares[i] += 1
                leftover -= float(prices[i])
                added = True
        if not added:
            return shares


def generate_picks(
    cfg: Dict[str, Any], out_dir: str, capital: Optional[float] = None
) -> pd.DataFrame:
    """Score the latest trading date and write the top-K picks with weights.

    ``capital`` (your current paper-account value) sizes each pick into dollars
    and shares; defaults to env.initial_cash. Every run also appends to the
    persistent paper-trading log at paper_trading/picks_log.csv.
    """
    rank = _rank_cfg(cfg)
    feature_set = str(rank["feature_set"])
    horizon = int(rank["label_horizon"])
    top_k = int(rank["top_k"])

    panel = load_universe_panel(cfg)
    featured = build_cross_features(panel, feature_set=feature_set)

    if str(rank["model"]) == "lambdarank":
        labeled = add_label_bins(
            add_forward_returns(featured, horizon=horizon), n_bins=int(rank["label_bins"])
        )
        scorer = LambdaRankScorer(
            feature_columns=model_feature_columns(feature_set),
            seed=int(cfg.get("training", {}).get("seed", 42)),
        ).fit(labeled)
        featured["score"] = scorer.score(featured)
    else:
        featured["score"] = featured[rank_col(str(rank["score_feature"]))]

    as_of = featured["date"].max()
    latest = featured[featured["date"] == as_of].set_index("ticker")
    picks = latest.nlargest(top_k, "score")

    rules = PortfolioRules(top_k=top_k, weighting=str(rank["weighting"]))
    weights = _target_weights(
        picks.index, list(picks.index), picks["vol_60"], rules, scale=1.0
    )

    cash = float(capital) if capital is not None else float(
        cfg.get("env", {}).get("initial_cash", 10_000.0)
    )
    w = weights.reindex(picks.index).to_numpy()
    closes = picks["close"].to_numpy()
    shares = _whole_share_allocation(w, closes, cash)
    cost = (shares * closes).round(2)

    out = pd.DataFrame(
        {
            "as_of": str(pd.Timestamp(as_of).date()),
            "rank": range(1, len(picks) + 1),
            "ticker": picks.index,
            "name": [name_of(t) for t in picks.index],
            "score": picks["score"].to_numpy(),
            "weight": w,
            "close": closes,
            "shares": shares,
            "cost": cost,
        }
    )

    path = Path(out_dir) / PICKS_FILENAME
    out.to_csv(path, index=False)
    _append_paper_log(out, PAPER_LOG_PATH)

    logger.info(
        f"Picks as of {out['as_of'].iloc[0]} (model={rank['model']}, top_k={top_k}, capital=${cash:,.0f}):"
    )
    for row in out.itertuples(index=False):
        logger.info(
            f"  #{row.rank:02d} {row.ticker:6s} {row.name:<28s} weight={row.weight:.3f} "
            f"close={row.close:9.2f}  buy {row.shares:>4d} sh = ${row.cost:>9,.2f}"
        )
    invested = float(out["cost"].sum())
    logger.info(
        f"Total invested: ${invested:,.2f} of ${cash:,.2f} (uninvested cash: ${cash - invested:,.2f})"
    )
    logger.info(f"Saved picks: {path} | appended to {PAPER_LOG_PATH}")
    return out
