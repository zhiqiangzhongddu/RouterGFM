"""META-DES meta-classifier lambda (Cruz et al., 2015, Sec. 3.2.3).

Paper: MLP with 10 hidden neurons trained with Levenberg-Marquardt on a random
75/25 train/validation split, stopping when validation performance has not
improved for 5 epochs. Here: tanh hidden layer (paper silent; MATLAB default),
sum-of-squares (MSE) objective, and full-batch L-BFGS with a strong-Wolfe line
search standing in for Levenberg-Marquardt. As with LM, an epoch is a single
second-order update, so early stopping acts at that granularity (several
updates per epoch already overfit few-shot meta-sets in the first epoch). The
best validation state is restored.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LBFGS_ITERS = 1  # one quasi-Newton update per epoch, like one Levenberg-Marquardt epoch


class MetaSelectorMLP(nn.Module):
    """``in -> hidden -> tanh -> 1 -> sigmoid``: probability that an expert is competent."""

    def __init__(self, in_dim: int, hidden: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(in_dim), int(hidden)), nn.Tanh(), nn.Linear(int(hidden), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def fit_meta_selector(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden: int,
    val_frac: float,
    patience: int,
    max_epochs: int,
    seed: int,
    device=None,
) -> Tuple[MetaSelectorMLP, Dict[str, Any]]:
    """Fit lambda on meta-features ``X [N, D]`` / meta-labels ``y [N]``.

    Returns the model (CPU, eval mode, best validation state) and
    ``{'val_mse', 'best_epoch', 'epochs'}``.
    """
    n = int(X.size(0))
    if n < 2:
        raise ValueError(f"The META-DES meta-training set needs at least two rows (got {n}).")
    device = torch.device(device) if device is not None else torch.device("cpu")
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator)
    n_val = min(max(int(round(float(val_frac) * n)), 1), n - 1)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X, y = X.float(), y.float()
    x_tr, y_tr = X[train_idx].to(device), y[train_idx].to(device)
    x_val, y_val = X[val_idx].to(device), y[val_idx].to(device)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = MetaSelectorMLP(X.size(1), hidden)
    model.to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=_LBFGS_ITERS, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.mse_loss(model(x_tr), y_tr)
        loss.backward()
        return loss

    best_mse, best_state, best_epoch, stale, epoch = float("inf"), copy.deepcopy(model.state_dict()), 0, 0, 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.step(closure)
        model.eval()
        with torch.no_grad():
            val_mse = float(F.mse_loss(model(x_val), y_val))
        if val_mse < best_mse:
            best_mse, best_state, best_epoch, stale = val_mse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    model.load_state_dict(best_state)
    model.cpu().eval()
    model.requires_grad_(False)
    return model, {"val_mse": best_mse, "best_epoch": best_epoch, "epochs": epoch}


__all__ = ["MetaSelectorMLP", "fit_meta_selector"]
