"""Training loop for the Double DQN agent."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ai_trader.agents import build_agent
from ai_trader.env import make_env_bundle
from ai_trader.utils import get_logger
from ai_trader.viz import plot_training_curves

from .evaluate import evaluate_policy, run_episode
from .logger import DQNTrainingLogger

logger = get_logger(__name__)


def train(cfg: Dict[str, Any], out_dir: str) -> None:
    train_env, val_env, _ = make_env_bundle(cfg)
    agent = build_agent(
        cfg=cfg,
        state_dim=int(train_env.observation_space.shape[0]),
        action_dim=int(train_env.action_space.n),
    )
    train_on_envs(cfg, train_env, val_env, agent, out_dir)
    train_env.close()
    val_env.close()


def train_on_envs(cfg: Dict[str, Any], train_env, val_env, agent, out_dir: str) -> str:
    """Run the DQN training loop on pre-built envs/agent; return the best-checkpoint path."""
    seed = int(cfg["training"]["seed"])
    episodes = int(cfg["training"]["episodes"])
    max_steps = int(cfg["training"]["max_steps_per_episode"])
    eval_every = int(cfg["training"].get("eval_every", 50))
    eval_episodes = int(cfg["training"].get("eval_episodes", 3))
    patience = int(cfg["training"].get("early_stop_patience", 0))
    warmup = int(cfg["training"].get("early_stop_warmup", 0))

    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metric_logger = DQNTrainingLogger()
    best_val_reward = -np.inf
    no_improve_count = 0
    patience_active = warmup == 0  # False during warmup; flips True at warmup end + resets bar
    best_path = ckpt_dir / "double_dqn_best.pt"
    best_saved = False

    logger.info("=== Training Start (Double DQN) ===")
    logger.info(
        "Observation dim: %d, Action dim: %d",
        train_env.observation_space.shape[0],
        train_env.action_space.n,
    )
    logger.info(
        "Episodes: %d  |  Early-stop patience: %s  |  Warmup: %s",
        episodes, patience or "off", warmup or "none",
    )

    start_time = time.time()
    for ep in range(1, episodes + 1):
        stats = run_episode(train_env, agent, train=True, max_steps=max_steps, seed=seed + ep)
        epsilon = agent.end_episode()

        logger.info(
            f"[Episode {ep:04d}] reward={stats['episode_reward']:.5f} "
            f"steps={stats['episode_steps']:03d} loss={stats['average_loss']:.6f} "
            f"eps={epsilon:.4f}"
        )

        eval_reward = None
        if eval_every > 0 and ep % eval_every == 0:
            eval_metrics = evaluate_policy(
                val_env, agent, episodes=eval_episodes, max_steps=max_steps,
                seed=seed + 100_000 + ep,
            )
            eval_reward = eval_metrics["avg_sharpe"]
            logger.info(
                f"  [Val] reward={eval_metrics['avg_reward']:.4f} "
                f"return={eval_metrics['avg_total_return']:.2%} "
                f"sharpe={eval_metrics['avg_sharpe']:.2f} "
                f"sortino={eval_metrics['avg_sortino']:.2f} "
                f"win%={eval_metrics['avg_win_rate']:.1%} "
                f"drawdown={eval_metrics['avg_max_drawdown']:.2%}"
            )

            latest_ckpt = ckpt_dir / "double_dqn_latest.pt"
            agent.save(str(latest_ckpt))

            if not patience_active and ep >= warmup:
                patience_active = True
                best_val_reward = -np.inf  # reset bar so post-warmup Sharpe is measured fresh
                logger.info(f"  Warmup complete at ep {ep} — patience counter active (metric: Sharpe).")

            if eval_reward > best_val_reward:
                best_val_reward = eval_reward
                agent.save(str(best_path))
                best_saved = True
                logger.info(f"  New best checkpoint (sharpe={eval_reward:.4f}): {best_path}")
                no_improve_count = 0
            elif patience > 0 and patience_active:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(
                        f"  Early stop: val Sharpe hasn't improved in "
                        f"{patience} eval intervals ({patience * eval_every} episodes)."
                    )
                    metric_logger.log_episode(
                        episode_reward=stats["episode_reward"],
                        episode_length=stats["episode_steps"],
                        average_loss=stats["average_loss"],
                        epsilon=epsilon,
                        eval_reward=eval_reward,
                    )
                    metric_logger.save(out_dir)
                    break

        metric_logger.log_episode(
            episode_reward=stats["episode_reward"],
            episode_length=stats["episode_steps"],
            average_loss=stats["average_loss"],
            epsilon=epsilon,
            eval_reward=eval_reward,
        )

        if (eval_every > 0 and ep % eval_every == 0) or ep == episodes:
            metric_logger.save(out_dir)

    elapsed = time.time() - start_time
    logger.info(f"=== Training Complete ({elapsed:.1f}s) ===")

    # Guarantee a usable best checkpoint even when evaluation/early-stop is disabled.
    if not best_saved:
        agent.save(str(best_path))

    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_training_curves(
        metric_logger.episode_rewards,
        metric_logger.episode_losses,
        plots_dir,
        eval_rewards=metric_logger.eval_rewards,
        epsilons=metric_logger.epsilons,
        episode_lengths=metric_logger.episode_lengths,
    )

    return str(best_path)
