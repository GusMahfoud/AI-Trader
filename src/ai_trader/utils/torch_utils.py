"""Torch device selection, tensor conversion, and target-network updates."""

from __future__ import annotations

import numpy as np
import torch


def get_device(request: str = "auto") -> torch.device:
    """Resolve "auto" / "cuda" / "cpu" against actual hardware availability."""
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError(
            "device='cuda' requested but torch.cuda.is_available() is False. "
            "Recreate the conda env from environment.yml (which pins pytorch-cuda=12.1), "
            "verify your NVIDIA driver is >= 528.33 on Windows, "
            "or set device: 'cpu' (or 'auto') in config.yaml to allow CPU fallback."
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(array, device: torch.device) -> torch.Tensor:
    if isinstance(array, list):
        array = np.array(array)
    return torch.tensor(array, dtype=torch.float32).to(device)


def hard_update(target_net: torch.nn.Module, source_net: torch.nn.Module) -> None:
    target_net.load_state_dict(source_net.state_dict())


def soft_update(target_net: torch.nn.Module, source_net: torch.nn.Module, tau: float) -> None:
    """Polyak averaging: θ_target ← τ·θ_source + (1−τ)·θ_target."""
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)
