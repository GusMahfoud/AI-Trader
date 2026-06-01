"""Episode-level metric logger writing CSVs to the output directory."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


PathLike = Union[str, Path]


@dataclass
class DQNTrainingLogger:
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_losses: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    eval_rewards: List[Optional[float]] = field(default_factory=list)

    def log_episode(
        self,
        episode_reward: float,
        episode_length: int,
        average_loss: float,
        epsilon: float,
        eval_reward: Optional[float] = None,
    ) -> None:
        self.episode_rewards.append(float(episode_reward))
        self.episode_lengths.append(int(episode_length))
        self.episode_losses.append(float(average_loss))
        self.epsilons.append(float(epsilon))
        self.eval_rewards.append(eval_reward)

    def save(self, output_dir: PathLike) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._write_metric(output_path / "episode_rewards.csv", "episode_reward", self.episode_rewards)
        self._write_metric(output_path / "episode_losses.csv", "episode_loss", self.episode_losses)
        self._write_metric(output_path / "episode_lengths.csv", "episode_length", self.episode_lengths)
        self._write_metric(output_path / "episode_epsilons.csv", "epsilon", self.epsilons)

    @staticmethod
    def _write_metric(file_path: Path, column_name: str, values: List[float]) -> None:
        with file_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", column_name])
            for episode_index, value in enumerate(values, start=1):
                w.writerow([episode_index, value])
