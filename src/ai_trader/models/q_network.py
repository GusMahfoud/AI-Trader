"""MLP Q-network approximating Q(s, a) for discrete actions."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Feedforward Q-network mapping state vectors to per-action Q-values."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 256, 128]

        layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
