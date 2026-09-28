"""META-DES meta-training data (Cruz et al., 2015, Sec. 3.2.2, Table 1; DESlib ``METADES``).

For classifier ``e`` and sample ``x_j`` the meta-feature vector
``v_{e,j} = [f1 (K), f2 (K), f3, f4 (Kp), f5]`` holds

* f1: hard correctness of ``e`` on the K nearest DSEL samples ``theta_j``
  (region of competence, ascending distance);
* f2: posterior ``e`` assigns to the true class of each neighbour in ``theta_j``;
* f3: local accuracy of ``e`` over ``theta_j``;
* f4: correctness of ``e`` on the Kp DSEL samples ``phi_j`` whose output
  profiles are closest to ``x_j``'s;
* f5: confidence of ``e`` on ``x_j`` (max posterior, as in DESlib).

DSEL here is the support set with out-of-fold posteriors (every support item
is predicted by a head not trained on it), used leave-one-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def pool_consensus(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Consensus ``H(x) = max_l (1/M) sum_e 1[label_e(x) = l]`` of pool decisions ``labels [n, M]``."""
    labels = labels.long()
    counts = torch.zeros(labels.size(0), int(num_classes), device=labels.device)
    counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=counts.dtype))
    return counts.max(dim=1).values / labels.size(1)


def knn_indices(query: torch.Tensor, ref: torch.Tensor, k: int, exclude: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Indices ``[n, k]`` of the k nearest ``ref`` rows (Euclidean), nearest first.

    ``exclude[i]`` is a ``ref`` index barred for query row ``i`` (leave-one-out
    by index, so duplicated points are still eligible neighbours).
    """
    dist = torch.cdist(query.float(), ref.float())
    if exclude is not None:
        dist[torch.arange(dist.size(0), device=dist.device), exclude.to(dist.device)] = float("inf")
    return dist.topk(int(k), dim=1, largest=False).indices


def compute_meta_features(
    correct_dsel: torch.Tensor,
    post_true_dsel: torch.Tensor,
    conf: torch.Tensor,
    idx_theta: torch.Tensor,
    idx_phi: torch.Tensor,
) -> torch.Tensor:
    """Meta-feature rows ``[n * M, 2K + Kp + 2]`` in (sample, expert) order.

    ``correct_dsel`` / ``post_true_dsel`` are ``[n_s, M]`` DSEL correctness and
    true-class posteriors, ``conf [n, M]`` the scored samples' confidences,
    ``idx_theta [n, K]`` / ``idx_phi [n, Kp]`` DSEL neighbour indices.
    """
    correct = correct_dsel.float()
    f1 = correct[idx_theta]  # [n, K, M]
    f2 = post_true_dsel.float()[idx_theta]  # [n, K, M]
    f3 = f1.mean(dim=1, keepdim=True)  # [n, 1, M]
    f4 = correct[idx_phi]  # [n, Kp, M]
    f5 = conf.float().unsqueeze(1)  # [n, 1, M]
    feats = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [n, D, M]
    return feats.transpose(1, 2).reshape(-1, feats.size(1))


@dataclass
class MetaTrainingSet:
    X: torch.Tensor  # [N_sel * M, D] meta-features
    y: torch.Tensor  # [N_sel * M] meta-labels alpha (1 = expert correct), float
    rows: torch.Tensor  # [N_sel] support indices used for meta-training
    theta: torch.Tensor  # [N_sel, K] leave-one-out regions of competence
    phi: torch.Tensor  # [N_sel, Kp] leave-one-out output-profile neighbours
    correct: torch.Tensor  # [n_s, M] OOF correctness (bool)
    post_true: torch.Tensor  # [n_s, M] OOF true-class posteriors
    k: int  # effective K = min(K, n_s - 1), shared with the generalization phase
    kp: int
    consensus_kept_frac: float  # share of S_a with H < hc (before the fallback)
    fallback: bool  # True when every support sample was used instead


def build_meta_training_set(
    z_s: torch.Tensor, oof_post: torch.Tensor, y_s: torch.Tensor, k: int, kp: int, hc: float
) -> MetaTrainingSet:
    """Algorithm 1 on DSEL = T_lambda = S_a with OOF posteriors ``oof_post [n_s, M, L]``.

    Sample selection keeps ``x_j`` with pool consensus ``H(x_j) < hc`` (paper
    rule); all of S_a is used when fewer than two samples survive or their
    meta-labels are all equal (the paper is silent; needed at few shots).
    """
    n_s, _, num_classes = oof_post.shape
    if n_s < 2:
        raise ValueError(f"META-DES needs at least two support instances (got {n_s}).")
    y_s = y_s.long().view(-1)
    pred = oof_post.argmax(dim=-1)
    correct = pred == y_s[:, None]
    post_true = oof_post.gather(2, y_s.view(-1, 1, 1).expand(-1, oof_post.size(1), 1)).squeeze(2)
    conf = oof_post.max(dim=-1).values

    keep = pool_consensus(pred, num_classes) < float(hc)
    rows = torch.nonzero(keep, as_tuple=False).view(-1)
    labels = correct[rows]
    fallback = rows.numel() < 2 or bool(labels.all()) or not bool(labels.any())
    if fallback:
        rows = torch.arange(n_s)

    k_eff, kp_eff = min(int(k), n_s - 1), min(int(kp), n_s - 1)
    profiles = oof_post.reshape(n_s, -1)
    theta = knn_indices(z_s[rows], z_s, k_eff, exclude=rows)
    phi = knn_indices(profiles[rows], profiles, kp_eff, exclude=rows)
    X = compute_meta_features(correct, post_true, conf[rows], theta, phi)
    return MetaTrainingSet(
        X=X,
        y=correct[rows].reshape(-1).float(),
        rows=rows,
        theta=theta,
        phi=phi,
        correct=correct,
        post_true=post_true,
        k=k_eff,
        kp=kp_eff,
        consensus_kept_frac=float(keep.float().mean()),
        fallback=fallback,
    )


__all__ = [
    "MetaTrainingSet",
    "build_meta_training_set",
    "compute_meta_features",
    "knn_indices",
    "pool_consensus",
]
