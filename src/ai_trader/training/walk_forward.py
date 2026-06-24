"""Walk-forward validation: retrain on each rolling window, aggregate test metrics.

Reports mean ± std of test-split performance across K folds so a single lucky
test window can't masquerade as a real edge.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ai_trader.agents import build_agent
from ai_trader.data import feature_columns, load_featured_frame, normalize_bundle
from ai_trader.data.splits import walk_forward_bounds
from ai_trader.env import make_env
from ai_trader.utils import get_logger

from .evaluate import evaluate_buy_and_hold_policy, evaluate_policy
from .train import train_on_envs

logger = get_logger(__name__)


def walk_forward(cfg: Dict[str, Any], out_dir: str) -> None:
    env_cfg = cfg.get("env", {})
    frame = load_featured_frame(cfg)
    columns = feature_columns(env_cfg)

    n_splits = int(cfg["training"].get("n_splits", 4))
    folds = walk_forward_bounds(
        length=len(frame),
        n_splits=n_splits,
        train_ratio=float(env_cfg.get("train_ratio", 0.7)),
        val_ratio=float(env_cfg.get("val_ratio", 0.15)),
    )

    max_steps = int(cfg["training"]["max_steps_per_episode"])
    eval_episodes = int(cfg["training"].get("eval_episodes", 10))
    seed = int(cfg["training"]["seed"])

    agent_rows: List[Dict[str, float]] = []
    bh_rows: List[Dict[str, float]] = []
    for i, bounds in enumerate(folds):
        bundle = normalize_bundle(frame, bounds, columns)
        fold_dir = Path(out_dir) / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_env = make_env(cfg, "train", bundle)
        val_env = make_env(cfg, "val", bundle)
        test_env = make_env(cfg, "test", bundle)
        agent = build_agent(
            cfg=cfg,
            state_dim=int(train_env.observation_space.shape[0]),
            action_dim=int(train_env.action_space.n),
        )

        logger.info(f"=== Walk-forward fold {i + 1}/{n_splits} | test rows {bounds['test']} ===")
        best_path = train_on_envs(cfg, train_env, val_env, agent, str(fold_dir))
        agent.load(best_path)

        a = evaluate_policy(test_env, agent, episodes=eval_episodes, max_steps=max_steps, seed=seed)
        b = evaluate_buy_and_hold_policy(test_env, episodes=eval_episodes, max_steps=max_steps, seed=seed)
        a["fold"] = float(i)
        b["fold"] = float(i)
        agent_rows.append(a)
        bh_rows.append(b)

        for env in (train_env, val_env, test_env):
            env.close()

    _write_summary(Path(out_dir), agent_rows, bh_rows)


def _agg(rows: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    vals = [r[key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def _write_summary(
    out_dir: Path, agent_rows: List[Dict[str, float]], bh_rows: List[Dict[str, float]]
) -> None:
    metric_keys = [k for k in agent_rows[0].keys() if k != "fold"]
    fieldnames = ["agent", "fold", *metric_keys]

    csv_path = out_dir / "walk_forward_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agent_rows:
            writer.writerow({"agent": "double_dqn", **row})
        for row in bh_rows:
            writer.writerow({"agent": "buy_and_hold", **row})
    logger.info(f"Saved walk-forward metrics: {csv_path}")

    _write_aggregate(out_dir, metric_keys, agent_rows, bh_rows)

    logger.info("=== Walk-Forward Summary (mean ± std across folds) ===")
    for label, rows in [("DoubleDQN", agent_rows), ("BuyAndHold", bh_rows)]:
        sh_m, sh_s = _agg(rows, "avg_sharpe")
        ret_m, ret_s = _agg(rows, "avg_total_return")
        dd_m, dd_s = _agg(rows, "avg_max_drawdown")
        logger.info(
            f"{label:11s} sharpe={sh_m:.3f}±{sh_s:.3f} "
            f"return={ret_m:.3%}±{ret_s:.3%} drawdown={dd_m:.3%}±{dd_s:.3%}"
        )


def _write_aggregate(
    out_dir: Path,
    metric_keys: List[str],
    agent_rows: List[Dict[str, float]],
    bh_rows: List[Dict[str, float]],
) -> None:
    """Write per-agent mean/std of each metric across folds to walk_forward_summary.csv."""
    fieldnames = ["agent", "stat", *metric_keys]
    csv_path = out_dir / "walk_forward_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agent_name, rows in [("double_dqn", agent_rows), ("buy_and_hold", bh_rows)]:
            means = {k: _agg(rows, k)[0] for k in metric_keys}
            stds = {k: _agg(rows, k)[1] for k in metric_keys}
            writer.writerow({"agent": agent_name, "stat": "mean", **means})
            writer.writerow({"agent": agent_name, "stat": "std", **stds})
    logger.info(f"Saved walk-forward summary: {csv_path}")
