"""Q-network: forward shape over both single and batched inputs."""

from __future__ import annotations

import torch

from ai_trader.models import QNetwork


def test_forward_shape_batched():
    net = QNetwork(state_dim=12, action_dim=3)
    x = torch.zeros(16, 12)
    out = net(x)
    assert out.shape == (16, 3)


def test_forward_shape_single():
    net = QNetwork(state_dim=12, action_dim=3)
    x = torch.zeros(1, 12)
    out = net(x)
    assert out.shape == (1, 3)


def test_custom_hidden_layers():
    net = QNetwork(state_dim=4, action_dim=2, hidden_layers=[8, 4])
    out = net(torch.zeros(2, 4))
    assert out.shape == (2, 2)
