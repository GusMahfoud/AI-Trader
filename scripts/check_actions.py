"""Inspect the action distribution of a trained DDQN checkpoint."""

import _bootstrap  # noqa: F401

from collections import Counter
from pathlib import Path

from ai_trader.agents import build_agent
from ai_trader.env import make_env_bundle
from ai_trader.training import run_episode
from ai_trader.utils import load_config, set_seed


def main() -> None:
    cfg = load_config("config.yaml")
    set_seed(42)

    _, _, test_env = make_env_bundle(cfg)
    agent = build_agent(
        cfg,
        state_dim=int(test_env.observation_space.shape[0]),
        action_dim=int(test_env.action_space.n),
    )
    ckpt_path = Path(cfg.get("output_dir", "results/double_dqn")) / "double_dqn_best.pt"
    agent.load(str(ckpt_path))

    for seed in (42, 123, 456):
        stats = run_episode(test_env, agent, train=False, max_steps=252, seed=seed)
        actions = stats["final_info"].get("action_history", [])
        counts = Counter(actions)
        total_trades = counts.get(1, 0) + counts.get(2, 0)
        pv = stats["final_info"].get("portfolio_value", 10_000)
        ret = (pv / 10_000 - 1) * 100
        print(
            f"Seed {seed}: Hold={counts.get(0, 0)} Buy={counts.get(2, 0)} "
            f"Sell={counts.get(1, 0)} | Trades={total_trades}/{len(actions)} "
            f"| Final=${pv:,.0f} ({ret:+.2f}%)"
        )


if __name__ == "__main__":
    main()
