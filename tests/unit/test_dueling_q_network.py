"""Tests for the Dueling Q-network architecture."""

from __future__ import annotations

import torch
import pytest

from ai_trader.models import DuelingQNetwork


@pytest.fixture
def net():
    return DuelingQNetwork(state_dim=16, action_dim=3)


def test_output_shape(net):
    x = torch.zeros(8, 16)
    out = net(x)
    assert out.shape == (8, 3)


def test_single_sample(net):
    x = torch.zeros(1, 16)
    out = net(x)
    assert out.shape == (1, 3)


def test_forward_equals_value_plus_centered_advantage(net):
    # Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
    x = torch.randn(32, 16)
    with torch.no_grad():
        features = net.shared(x)
        value = net.value_head(features)
        advantage = net.advantage_head(features)
        expected = value + advantage - advantage.mean(dim=1, keepdim=True)
        actual = net(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_different_from_mlp():
    from ai_trader.models import QNetwork
    torch.manual_seed(0)
    dueling = DuelingQNetwork(state_dim=16, action_dim=3)
    mlp = QNetwork(state_dim=16, action_dim=3)
    x = torch.randn(4, 16)
    with torch.no_grad():
        assert not torch.allclose(dueling(x), mlp(x)), "Dueling and MLP should differ"


def test_gradient_flows(net):
    x = torch.randn(4, 16)
    loss = net(x).sum()
    loss.backward()
    for p in net.parameters():
        assert p.grad is not None
