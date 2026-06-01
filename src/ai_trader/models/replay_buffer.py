"""Pre-allocated numpy ring buffer for experience replay."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-capacity ring buffer over contiguous numpy arrays.

    Faster than a deque of Python dataclasses because sampling is a single
    fancy-indexing pass instead of per-item attribute access.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self._size = 0
        self._ptr = 0

        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._states[self._ptr] = np.asarray(state, dtype=np.float32).reshape(-1)
        self._actions[self._ptr] = int(action)
        self._rewards[self._ptr] = float(reward)
        self._next_states[self._ptr] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self._dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if batch_size > self._size:
            raise ValueError("Not enough samples in buffer.")

        idxs = np.random.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[idxs],
            self._actions[idxs],
            self._rewards[idxs],
            self._next_states[idxs],
            self._dones[idxs],
        )

    def __len__(self) -> int:
        return self._size
