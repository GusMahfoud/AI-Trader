"""Live inference: train on all available history, emit today's top-K picks.

This is the paper-trading entry point (`--mode picks`): no folds, no backtest —
it fits the configured scorer on every labeled row up to the present, scores
the most recent trading date, and writes the ranked portfolio (with weights)
to picks.csv. Deliberately CLI-only: it is an interactive read of the model,
not an experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ai_trader.data.cross_features import build_cross_features, model_feature_columns, rank_col
from ai_trader.data.labels import add_forward_returns, add_label_bins
from ai_trader.data.universe import load_universe_panel
from ai_trader.models.ranker import LambdaRankScorer
from ai_trader.utils import get_logger

from .portfolio_sim import PortfolioRules, _target_weights
from .rank_backtest import _rank_cfg

logger = get_logger(__name__)

PICKS_FILENAME = "picks.csv"


def generate_picks(cfg: Dict[str, Any], out_dir: str) -> pd.DataFrame:
    """Score the latest trading date and write the top-K picks with weights."""
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

    out = pd.DataFrame(
        {
            "as_of": str(pd.Timestamp(as_of).date()),
            "rank": range(1, len(picks) + 1),
            "ticker": picks.index,
            "score": picks["score"].to_numpy(),
            "weight": weights.reindex(picks.index).to_numpy(),
            "close": picks["close"].to_numpy(),
        }
    )

    path = Path(out_dir) / PICKS_FILENAME
    out.to_csv(path, index=False)
    logger.info(f"Picks as of {out['as_of'].iloc[0]} (model={rank['model']}, top_k={top_k}):")
    for row in out.itertuples(index=False):
        logger.info(f"  #{row.rank:02d} {row.ticker:6s} weight={row.weight:.3f} close={row.close:.2f}")
    logger.info(f"Saved picks: {path}")
    return out
