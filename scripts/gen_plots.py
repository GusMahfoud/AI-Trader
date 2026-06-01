"""Regenerate training plots from saved CSVs (no retraining required)."""

import _bootstrap  # noqa: F401

import csv
from pathlib import Path

from ai_trader.utils import load_config
from ai_trader.viz import plot_training_curves


def _read_col(path: Path, col: str):
    with path.open(encoding="utf-8") as f:
        return [float(row[col]) for row in csv.DictReader(f)]


def main() -> None:
    cfg = load_config("config.yaml")
    results_dir = Path(cfg.get("output_dir", "results/double_dqn"))
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir.resolve()}")

    rewards = _read_col(results_dir / "episode_rewards.csv", "episode_reward")
    losses = _read_col(results_dir / "episode_losses.csv", "episode_loss")
    epsilons = _read_col(results_dir / "episode_epsilons.csv", "epsilon")
    lengths = _read_col(results_dir / "episode_lengths.csv", "episode_length")

    plot_training_curves(
        rewards=rewards,
        losses=losses,
        output_dir=results_dir,
        epsilons=epsilons,
        episode_lengths=[int(x) for x in lengths],
    )
    print(f"Saved plots to {results_dir.resolve()}")


if __name__ == "__main__":
    main()
