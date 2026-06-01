"""Random seeding for reproducible runs."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # RL is highly stochastic; deterministic kernels make debugging tractable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
