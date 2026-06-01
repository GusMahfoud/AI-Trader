"""Replay buffer: add/sample roundtrip, capacity wrap, error paths."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trader.models import ReplayBuffer


def test_sample_shapes_and_roundtrip():
    buf = ReplayBuffer(capacity=100, state_dim=4)
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.normal(size=4).astype(np.float32)
        sp = rng.normal(size=4).astype(np.float32)
        buf.add(s, action=1, reward=0.5, next_state=sp, done=False)

    assert len(buf) == 50

    states, actions, rewards, next_states, dones = buf.sample(batch_size=8)
    assert states.shape == (8, 4)
    assert next_states.shape == (8, 4)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)
    assert states.dtype == np.float32
    assert actions.dtype == np.int64


def test_capacity_wraps_in_ring_order():
    buf = ReplayBuffer(capacity=3, state_dim=2)
    for i in range(5):
        buf.add(
            state=np.array([i, i], dtype=np.float32),
            action=i,
            reward=float(i),
            next_state=np.array([i + 1, i + 1], dtype=np.float32),
            done=False,
        )

    assert len(buf) == 3
    # Ring buffer holds the three most-recent transitions (i=2, 3, 4).
    stored_actions = sorted(int(a) for a in buf._actions)
    assert stored_actions == [2, 3, 4]


def test_sample_requires_enough_samples():
    buf = ReplayBuffer(capacity=10, state_dim=2)
    buf.add(np.zeros(2, np.float32), 0, 0.0, np.zeros(2, np.float32), False)
    with pytest.raises(ValueError):
        buf.sample(batch_size=5)


def test_sample_rejects_invalid_batch_size():
    buf = ReplayBuffer(capacity=10, state_dim=2)
    buf.add(np.zeros(2, np.float32), 0, 0.0, np.zeros(2, np.float32), False)
    with pytest.raises(ValueError):
        buf.sample(batch_size=0)
