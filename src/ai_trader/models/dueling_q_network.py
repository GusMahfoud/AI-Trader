"""Dueling Q-network: separates state value V(s) from advantage A(s,a)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a)).

    Decouples learning the value of being in a state from the relative
    advantage of each action — improves stability in states where the
    choice of action has little effect (e.g. flat market, no open position).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 256]

        shared: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_layers:
            shared.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.shared = nn.Sequential(*shared)

        self.value_head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        value = self.value_head(features)            # (B, 1)
        advantage = self.advantage_head(features)    # (B, action_dim)
        # Mean subtraction enforces identifiability of V and A.
        return value + advantage - advantage.mean(dim=1, keepdim=True)
