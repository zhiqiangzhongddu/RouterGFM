"""Utility functions for data loading and model weight loading."""

import torch


def safe_torch_load(path):
    """Compatibility loader for Torch 2.6+ weights-only default."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)
    