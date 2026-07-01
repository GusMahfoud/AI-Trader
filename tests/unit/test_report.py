"""Walk-forward report artifact: schema, aggregates, verdict, and JSON round-trip."""

from __future__ import annotations

import json
from typing import Dict, List

import pytest

from ai_trader.training.report import (
    REPORT_SCHEMA_VERSION,
    build_walk_forward_report,
    write_report,
)


def _fold_rows(sharpes: List[float], returns: List[float]) -> List[Dict[str, float]]:
    return [
        {
            "fold": float(i),
            "avg_sharpe": s,
            "avg_total_return": r,
            "avg_max_drawdown": -0.1,
        }
        for i, (s, r) in enumerate(zip(sharpes, returns))
    ]


@pytest.fixture
def report():
    agent_rows = _fold_rows([1.0, 2.0, -0.5, 1.5], [0.10, 0.30, -0.05, 0.20])
    bh_rows = _fold_rows([0.5, 2.5, 0.5, 0.5], [0.05, 0.40, 0.05, 0.05])
    return build_walk_forward_report(
        run_id="wf_test", cfg={"training": {"seed": 42}}, agent_rows=agent_rows, bh_rows=bh_rows
    )


def test_schema_and_identity(report) -> None:
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["kind"] == "walk_forward"
    assert report["run_id"] == "wf_test"
    assert report["config"] == {"training": {"seed": 42}}
    assert set(report["agents"]) == {"double_dqn", "buy_and_hold"}


def test_aggregates(report) -> None:
    dqn = report["agents"]["double_dqn"]
    assert dqn["mean"]["avg_sharpe"] == pytest.approx(1.0)
    assert dqn["std"]["avg_sharpe"] == pytest.approx(0.93541, abs=1e-4)
    assert len(dqn["folds"]) == 4
    assert dqn["folds"][2]["avg_sharpe"] == -0.5


def test_verdict(report) -> None:
    verdict = report["verdict"]
    assert verdict["n_folds"] == 4
    assert verdict["sharpe_edge_vs_bh"] == pytest.approx(0.0)
    assert verdict["folds_beating_bh"] == 2
    assert verdict["folds_positive"] == 3
    assert verdict["beats_bh_sharpe"] is False


def test_mismatched_rows_rejected() -> None:
    rows = _fold_rows([1.0], [0.1])
    with pytest.raises(ValueError):
        build_walk_forward_report(run_id="x", cfg={}, agent_rows=rows, bh_rows=[])


def test_json_round_trip(report, tmp_path) -> None:
    path = write_report(tmp_path, report)
    assert path.name == "report.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["verdict"]["folds_beating_bh"] == 2
    assert loaded["agents"]["double_dqn"]["mean"]["avg_sharpe"] == pytest.approx(1.0)
