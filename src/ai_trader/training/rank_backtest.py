"""Cross-sectional rank backtest: score universe, hold top-K, compare to benchmarks.

Phase-3 baseline of the cross-sectional pivot: a heuristic momentum score (no
ML) proves the data pipeline can reproduce the documented momentum premium
before any model earns trust. Folds come from purged forward-only splits;
benchmarks are an equal-weight universe portfolio and single-ticker buy-and-hold
(SPY by default), all cost-inclusive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ai_trader.data.cross_features import build_cross_features, model_feature_columns, rank_col
from ai_trader.data.labels import LABEL_COL, add_forward_returns, add_label_bins
from ai_trader.data.loader import _load_market_data
from ai_trader.data.purged_splits import assert_no_label_overlap, purged_walk_forward_bounds
from ai_trader.data.universe import close_matrix, load_universe_panel
from ai_trader.models.ranker import LambdaRankScorer
from ai_trader.risk.metrics import equity_summary
from ai_trader.risk.signal_metrics import ic_series, ic_summary, ndcg_series, topk_spread_series
from ai_trader.utils import get_logger

from .portfolio_sim import PortfolioRules, simulate_rank_portfolio
from .report import build_rank_report, write_report

logger = get_logger(__name__)


def _rank_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rank = dict(cfg.get("rank", {}))
    rank.setdefault("top_k", 5)
    rank.setdefault("rebalance_days", 21)
    rank.setdefault("label_horizon", 21)
    rank.setdefault("test_ratio", 0.4)
    rank.setdefault("score_feature", "mom_12_1")
    rank.setdefault("benchmark_ticker", "SPY")
    rank.setdefault("model", "momentum")
    rank.setdefault("label_bins", 4)
    rank.setdefault("feature_set", "v1")
    rank.setdefault("buffer_k", 0)
    rank.setdefault("weighting", "equal")
    rank.setdefault("dd_brake", 0.0)
    rank.setdefault("sector_cap", 0)
    return rank


def _signal_row(test_panel: pd.DataFrame, top_k: int) -> Dict[str, float]:
    """IC/NDCG/top-K-spread of the score against realized forward returns."""
    labeled = test_panel.dropna(subset=[LABEL_COL, "score"])
    if labeled.empty:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_hit_rate": 0.0,
                "ndcg_at_k": 0.0, "topk_spread": 0.0}
    labeled = labeled.rename(columns={LABEL_COL: "fwd_return"})
    out = dict(ic_summary(ic_series(labeled)))
    out["ndcg_at_k"] = float(ndcg_series(labeled, k=top_k).mean())
    out["topk_spread"] = float(topk_spread_series(labeled, k=top_k).mean())
    return out


def _benchmark_equity(cfg: Dict[str, Any], ticker: str, dates: pd.DatetimeIndex, cost_rate: float) -> np.ndarray:
    """Cost-inclusive buy-and-hold equity for one benchmark ticker on given dates."""
    env_cfg = {**cfg.get("env", {}), "ticker": ticker, "universe": [], "auto_adjust": True}
    frame = _load_market_data(env_cfg, seed=int(cfg.get("training", {}).get("seed", 42)))
    closes = frame.set_index("date")["close"].reindex(dates).ffill().bfill()
    equity = (closes / closes.iloc[0]).to_numpy()
    return equity * (1.0 - cost_rate)  # entry cost on the full notional


def rank_backtest(cfg: Dict[str, Any], out_dir: str) -> None:
    rank = _rank_cfg(cfg)
    env_cfg = cfg.get("env", {})
    cost_rate = float(env_cfg.get("transaction_cost", 0.0003)) + float(env_cfg.get("slippage", 0.0001))
    score_column = rank_col(str(rank["score_feature"]))
    top_k = int(rank["top_k"])
    horizon = int(rank["label_horizon"])

    model = str(rank["model"])
    feature_set = str(rank["feature_set"])
    seed = int(cfg.get("training", {}).get("seed", 42))

    panel = load_universe_panel(cfg)
    featured = add_forward_returns(
        build_cross_features(panel, feature_set=feature_set), horizon=horizon
    )
    if model == "lambdarank":
        featured = add_label_bins(featured, n_bins=int(rank["label_bins"]))
    closes = close_matrix(panel)

    dates = pd.DatetimeIndex(featured["date"].unique()).sort_values()
    folds = purged_walk_forward_bounds(
        n_dates=len(dates),
        n_splits=int(cfg.get("training", {}).get("n_splits", 4)),
        label_horizon=horizon,
        test_ratio=float(rank["test_ratio"]),
    )
    assert_no_label_overlap(folds, label_horizon=horizon)

    strategy_rows: List[Dict[str, float]] = []
    benchmark_rows: Dict[str, List[Dict[str, float]]] = {"equal_weight": [], "benchmark_bh": []}
    bench_ticker = str(rank["benchmark_ticker"])

    for k, fold in enumerate(folds):
        lo, hi = fold["test"]
        test_dates = dates[lo:hi]
        test_closes = closes.reindex(test_dates)
        test_panel = featured[featured["date"].isin(test_dates)].copy()

        if model == "lambdarank":
            train_dates = dates[fold["train"][0] : fold["train"][1]]
            train_panel = featured[featured["date"].isin(train_dates)]
            scorer = LambdaRankScorer(
                feature_columns=model_feature_columns(feature_set), seed=seed
            ).fit(train_panel)
            test_panel["score"] = scorer.score(test_panel)
        else:
            test_panel["score"] = test_panel[score_column]

        score_cols = test_panel[["date", "ticker", "score", "vol_60"]].rename(
            columns={"vol_60": "vol"}
        )
        strat_rules = PortfolioRules(
            top_k=top_k,
            rebalance_days=int(rank["rebalance_days"]),
            cost_rate=cost_rate,
            buffer_k=int(rank["buffer_k"]),
            weighting=str(rank["weighting"]),
            dd_brake=float(rank["dd_brake"]),
            sector_cap=int(rank["sector_cap"]),
        )
        ew_rules = PortfolioRules(
            top_k=None, rebalance_days=int(rank["rebalance_days"]), cost_rate=cost_rate
        )
        strat_equity = simulate_rank_portfolio(test_closes, score_cols, strat_rules)
        ew_equity = simulate_rank_portfolio(test_closes, None, ew_rules)
        bench_equity = _benchmark_equity(cfg, bench_ticker, test_dates, cost_rate)

        row = {**equity_summary(strat_equity), **_signal_row(test_panel, top_k), "fold": float(k)}
        strategy_rows.append(row)
        benchmark_rows["equal_weight"].append({**equity_summary(ew_equity), "fold": float(k)})
        benchmark_rows["benchmark_bh"].append({**equity_summary(bench_equity), "fold": float(k)})

        logger.info(
            f"[fold {k + 1}/{len(folds)}] strat sharpe={row['sharpe']:.3f} "
            f"ret={row['total_return']:.2%} ic={row['ic_mean']:.4f} | "
            f"ew sharpe={benchmark_rows['equal_weight'][-1]['sharpe']:.3f} | "
            f"{bench_ticker} sharpe={benchmark_rows['benchmark_bh'][-1]['sharpe']:.3f}"
        )

    report = build_rank_report(
        run_id=Path(out_dir).name,
        cfg=cfg,
        strategy_rows=strategy_rows,
        benchmark_rows=benchmark_rows,
    )
    path = write_report(Path(out_dir), report)
    logger.info(f"Saved rank backtest report: {path}")
