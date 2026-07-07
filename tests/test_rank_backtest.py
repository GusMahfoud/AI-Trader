"""Integration test: full rank backtest on a synthetic universe, offline."""

from __future__ import annotations

import json

import pytest

from ai_trader.training.rank_backtest import rank_backtest


@pytest.fixture
def rank_config(synthetic_config):
    cfg = {**synthetic_config}
    cfg["env"] = {
        **synthetic_config["env"],
        "synthetic_length": 900,
        "universe": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
    }
    cfg["training"] = {**synthetic_config["training"], "n_splits": 2}
    cfg["rank"] = {
        "top_k": 2,
        "rebalance_days": 10,
        "label_horizon": 10,
        "test_ratio": 0.4,
        "score_feature": "mom_12_1",
        "benchmark_ticker": "SPY",
    }
    return cfg


def test_rank_backtest_produces_report(rank_config, tmp_path):
    rank_backtest(rank_config, out_dir=str(tmp_path))

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["kind"] == "rank_backtest"
    assert set(report["agents"]) == {"rank_strategy", "equal_weight", "benchmark_bh"}

    strat = report["agents"]["rank_strategy"]
    assert len(strat["folds"]) == 2
    for row in strat["folds"]:
        for key in ("sharpe", "total_return", "max_drawdown", "ic_mean", "ndcg_at_k", "topk_spread"):
            assert key in row
            assert abs(float(row[key])) < 1e6

    verdict = report["verdict"]
    assert verdict["n_folds"] == 2
    assert "sharpe_edge_vs_equal_weight" in verdict
    assert "sharpe_edge_vs_benchmark_bh" in verdict
