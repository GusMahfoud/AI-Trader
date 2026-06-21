"""Prioritized Experience Replay buffer using a sum-tree for O(log n) sampling."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class _SumTree:
    """Binary sum-tree: leaves hold priorities, internal nodes hold subtree sums."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._ptr = 0
        self._size = 0

    def _propagate(self, idx: int, delta: float) -> None:
        while idx > 0:
            idx = (idx - 1) // 2
            self._tree[idx] += delta

    def update(self, leaf_idx: int, priority: float) -> None:
        tree_idx = leaf_idx + self.capacity - 1
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def add(self, priority: float) -> int:
        idx = self._ptr
        self.update(idx, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return idx

    def get(self, value: float) -> Tuple[int, float]:
        """Walk tree to find leaf whose cumulative range contains *value*."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, float(self._tree[idx])

    @property
    def total(self) -> float:
        return float(self._tree[0])

    @property
    def max_priority(self) -> float:
        if self._size == 0:
            return 1.0
        leaves = self._tree[self.capacity - 1: self.capacity - 1 + self._size]
        return float(leaves.max()) or 1.0

    def __len__(self) -> int:
        return self._size


class PrioritizedReplayBuffer:
    """Replay buffer that samples transitions proportional to TD-error priority.

    New transitions receive max priority so they are guaranteed to be trained
    on at least once.  After each train step the agent should call
    ``update_priorities`` with the fresh TD errors.
    """

    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6):
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha

        self._tree = _SumTree(capacity)
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done) -> None:
        priority = self._tree.max_priority ** self.alpha
        idx = self._tree.add(priority)
        self._states[idx] = np.asarray(state, dtype=np.float32).reshape(-1)
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self._dones[idx] = float(done)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self._tree)
        if batch_size > n:
            raise ValueError("Not enough samples in buffer.")

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        segment = self._tree.total / batch_size

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            indices[i], priorities[i] = self._tree.get(value)

        probs = priorities / (self._tree.total + 1e-8)
        # IS weights normalised so max weight = 1.
        weights = (n * probs + 1e-8) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._tree.update(int(idx), float(p))

    def __len__(self) -> int:
        return len(self._tree)
