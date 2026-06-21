"""Tests for the Prioritized Experience Replay buffer."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trader.models import PrioritizedReplayBuffer


STATE_DIM = 8
CAPACITY = 100


@pytest.fixture
def buf():
    return PrioritizedReplayBuffer(capacity=CAPACITY, state_dim=STATE_DIM, alpha=0.6)


def _add(buf, n=1, reward=0.0):
    for i in range(n):
        buf.add(
            state=np.zeros(STATE_DIM),
            action=0,
            reward=reward,
            next_state=np.zeros(STATE_DIM),
            done=False,
        )


def test_len_grows(buf):
    assert len(buf) == 0
    _add(buf, 10)
    assert len(buf) == 10


def test_len_caps_at_capacity(buf):
    _add(buf, CAPACITY + 50)
    assert len(buf) == CAPACITY


def test_sample_shapes(buf):
    _add(buf, 20)
    s, a, r, ns, d, idx, w = buf.sample(8, beta=0.4)
    assert s.shape == (8, STATE_DIM)
    assert a.shape == (8,)
    assert r.shape == (8,)
    assert ns.shape == (8, STATE_DIM)
    assert d.shape == (8,)
    assert idx.shape == (8,)
    assert w.shape == (8,)


def test_weights_in_range(buf):
    _add(buf, 20)
    *_, w = buf.sample(8, beta=0.4)
    assert np.all(w > 0)
    assert np.all(w <= 1.0 + 1e-6), f"max weight={w.max()}"


def test_update_priorities_changes_distribution(buf):
    _add(buf, 50)
    # Give first 10 transitions very high priority.
    for i in range(10):
        buf._tree.update(i, 1000.0)
    # Give the rest very low priority.
    for i in range(10, 50):
        buf._tree.update(i, 0.001)

    _, _, _, _, _, indices, _ = buf.sample(20, beta=0.4)
    high_priority_count = int(np.sum(indices < 10))
    assert high_priority_count >= 10, "High-priority transitions should dominate sampling"


def test_update_priorities_method(buf):
    _add(buf, 20)
    _, _, _, _, _, indices, _ = buf.sample(8, beta=0.4)
    td_errors = np.ones(8, dtype=np.float32) * 0.5
    buf.update_priorities(indices, td_errors)  # should not raise


def test_insufficient_samples_raises(buf):
    _add(buf, 5)
    with pytest.raises(ValueError):
        buf.sample(10)
