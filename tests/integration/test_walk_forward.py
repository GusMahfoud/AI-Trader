"""Integration test: walk-forward runs end-to-end on synthetic data and writes a summary."""

from __future__ import annotations

import csv

from ai_trader.training import walk_forward


def test_walk_forward_writes_summary(synthetic_config, tmp_path):
    cfg = synthetic_config
    cfg["training"]["n_splits"] = 2

    walk_forward(cfg, out_dir=str(tmp_path))

    summary = tmp_path / "walk_forward_test.csv"
    assert summary.exists()

    with summary.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # One double_dqn row + one buy_and_hold row per fold.
    agents = [r["agent"] for r in rows]
    assert agents.count("double_dqn") == 2
    assert agents.count("buy_and_hold") == 2
    assert {r["fold"] for r in rows} == {"0.0", "1.0"}
    assert "avg_sharpe" in rows[0]

    # Each fold trained and saved a best checkpoint.
    for i in range(2):
        assert (tmp_path / f"fold_{i}" / "checkpoints" / "double_dqn_best.pt").exists()
