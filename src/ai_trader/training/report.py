"""Machine-readable walk-forward report artifact (report.json).

The single JSON document the platform consumes for "validate my model": per-fold
metrics, cross-fold aggregates, and an explicit verdict vs buy-and-hold. CSVs
remain for humans; this file is the API contract.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPORT_SCHEMA_VERSION = 1
REPORT_FILENAME = "report.json"


def _mean_std(rows: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    vals = [float(r[key]) for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    """Fold rows -> {mean: {...}, std: {...}, folds: [...]} over every metric."""
    metric_keys = [k for k in rows[0] if k != "fold"]
    return {
        "mean": {k: _mean_std(rows, k)[0] for k in metric_keys},
        "std": {k: _mean_std(rows, k)[1] for k in metric_keys},
        "folds": [dict(r) for r in rows],
    }


def _verdict(
    agent_rows: List[Dict[str, float]], bh_rows: List[Dict[str, float]]
) -> Dict[str, Any]:
    """Headline comparison of the agent against buy-and-hold across folds."""
    agent_sharpe, _ = _mean_std(agent_rows, "avg_sharpe")
    bh_sharpe, _ = _mean_std(bh_rows, "avg_sharpe")
    by_fold = list(zip(agent_rows, bh_rows))
    return {
        "n_folds": len(agent_rows),
        "sharpe_edge_vs_bh": agent_sharpe - bh_sharpe,
        "return_edge_vs_bh": (
            _mean_std(agent_rows, "avg_total_return")[0]
            - _mean_std(bh_rows, "avg_total_return")[0]
        ),
        "folds_beating_bh": sum(
            1 for a, b in by_fold if float(a["avg_sharpe"]) > float(b["avg_sharpe"])
        ),
        "folds_positive": sum(1 for a in agent_rows if float(a["avg_sharpe"]) > 0.0),
        "beats_bh_sharpe": agent_sharpe > bh_sharpe,
    }


def build_walk_forward_report(
    run_id: str,
    cfg: Dict[str, Any],
    agent_rows: List[Dict[str, float]],
    bh_rows: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Assemble the full walk-forward report from per-fold metric rows."""
    if not agent_rows or len(agent_rows) != len(bh_rows):
        raise ValueError(
            f"Need matching non-empty fold rows (got {len(agent_rows)} agent, "
            f"{len(bh_rows)} buy-and-hold)"
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "walk_forward",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "agents": {
            "double_dqn": _aggregate(agent_rows),
            "buy_and_hold": _aggregate(bh_rows),
        },
        "verdict": _verdict(agent_rows, bh_rows),
    }


def _rank_verdict(
    strategy_rows: List[Dict[str, float]],
    benchmark_rows: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Any]:
    """Headline comparison of the ranked portfolio against each benchmark."""
    strat_sharpe, _ = _mean_std(strategy_rows, "sharpe")
    verdict: Dict[str, Any] = {
        "n_folds": len(strategy_rows),
        "strategy_sharpe": strat_sharpe,
        "ic_mean": _mean_std(strategy_rows, "ic_mean")[0],
        "ic_ir": _mean_std(strategy_rows, "ic_ir")[0],
    }
    for name, rows in benchmark_rows.items():
        bench_sharpe, _ = _mean_std(rows, "sharpe")
        beats = sum(
            1 for s, b in zip(strategy_rows, rows)
            if float(s["sharpe"]) > float(b["sharpe"])
        )
        verdict[f"sharpe_edge_vs_{name}"] = strat_sharpe - bench_sharpe
        verdict[f"folds_beating_{name}"] = beats
        verdict[f"beats_{name}"] = strat_sharpe > bench_sharpe
    return verdict


def build_rank_report(
    run_id: str,
    cfg: Dict[str, Any],
    strategy_rows: List[Dict[str, float]],
    benchmark_rows: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Any]:
    """Assemble the rank-backtest report from per-fold strategy/benchmark rows."""
    if not strategy_rows:
        raise ValueError("Need at least one strategy fold row.")
    for name, rows in benchmark_rows.items():
        if len(rows) != len(strategy_rows):
            raise ValueError(f"Benchmark '{name}' has {len(rows)} rows, expected {len(strategy_rows)}.")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "rank_backtest",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "agents": {
            "rank_strategy": _aggregate(strategy_rows),
            **{name: _aggregate(rows) for name, rows in benchmark_rows.items()},
        },
        "verdict": _rank_verdict(strategy_rows, benchmark_rows),
    }


def write_report(out_dir: Path, report: Dict[str, Any]) -> Path:
    """Write the report to <out_dir>/report.json and return the path."""
    path = Path(out_dir) / REPORT_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    return path
