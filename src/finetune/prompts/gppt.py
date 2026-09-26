"""Prompt modules for GPPT-style finetuning."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.finetune.task_heads import prepare_single_label_labels
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional runtime dependency
    KMeans = None


class SimpleMeanConv(MessagePassing):
    """Neighborhood mean aggregation used by GPPT."""

    def __init__(self, add_self_loops_in_conv: bool = False):
        super().__init__(aggr="mean")
        self.add_self_loops_in_conv = bool(add_self_loops_in_conv)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.add_self_loops_in_conv:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index=edge_index, x=x, size=(x.size(0), x.size(0)))

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


def _run_kmeans_torch(
    x: Tensor,
    num_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    restarts: int = 1,
) -> Tensor:
    """Torch KMeans fallback when sklearn is unavailable."""
    if x.numel() == 0:
        return x.new_zeros((num_clusters, x.size(-1)))

    feat = torch.nan_to_num(x.detach()).float()
    n = feat.size(0)
    k = max(1, int(num_clusters))
    max_iter = max(1, int(max_iter))
    restarts = max(1, int(restarts))
    tol = float(max(tol, 0.0))

    if n <= 1:
        return feat[0:1].repeat(k, 1).to(dtype=x.dtype)
    if n < k:
        pad_idx = torch.randint(0, n, (k - n,), device=feat.device)
        feat = torch.cat([feat, feat[pad_idx]], dim=0)
        n = feat.size(0)

    best_centers = None
    best_inertia = None
    arange_n = torch.arange(n, device=feat.device)

    for _ in range(restarts):
        init_idx = torch.randperm(n, device=feat.device)[:k]
        centers = feat[init_idx].clone()

        for _ in range(max_iter):
            distances = torch.cdist(feat, centers, p=2)
            assign = distances.argmin(dim=1)

            new_centers = torch.zeros_like(centers)
            new_centers.index_add_(0, assign, feat)
            counts = torch.bincount(assign, minlength=k).to(feat.dtype).unsqueeze(1)
            non_empty = counts.view(-1) > 0

            if bool(non_empty.any().item()):
                new_centers[non_empty] = new_centers[non_empty] / counts[non_empty]
            if bool((~non_empty).any().item()):
                refill = torch.randint(0, n, (int((~non_empty).sum().item()),), device=feat.device)
                new_centers[~non_empty] = feat[refill]

            shift = torch.norm(new_centers - centers) / centers.norm().clamp_min(1e-12)
            centers = new_centers
            if float(shift.item()) <= tol:
                break

        final_dist = torch.cdist(feat, centers, p=2)
        final_assign = final_dist.argmin(dim=1)
        inertia = float((final_dist[arange_n, final_assign] ** 2).mean().item())
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers

    if best_centers is None:
        best_centers = feat[:k].clone()
    return best_centers.to(dtype=x.dtype)


def _run_kmeans_sklearn(
    x: Tensor,
    num_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 0,
    n_init: int = 10,
) -> Tensor:
    if x.numel() == 0:
        return x.new_zeros((num_clusters, x.size(-1)))

    feat = torch.nan_to_num(x.detach()).float()
    n = feat.size(0)
    k = max(1, int(num_clusters))
    if n <= 1:
        return feat[0:1].repeat(k, 1).to(dtype=x.dtype)
    if n < k:
        pad_idx = torch.randint(0, n, (k - n,), device=feat.device)
        feat = torch.cat([feat, feat[pad_idx]], dim=0)

    kmeans = KMeans(
        n_clusters=k,
        random_state=int(random_state),
        n_init=max(1, int(n_init)),
        max_iter=max(1, int(max_iter)),
        tol=float(max(tol, 0.0)),
    )
    centers_np = kmeans.fit(feat.cpu().numpy()).cluster_centers_
    centers = torch.from_numpy(centers_np).to(device=x.device, dtype=x.dtype)
    return centers


def _run_kmeans(
    x: Tensor,
    num_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    restarts: int = 1,
    use_sklearn: bool = True,
    random_state: int = 0,
    n_init: int = 10,
) -> Tensor:
    if bool(use_sklearn) and KMeans is not None:
        try:
            return _run_kmeans_sklearn(
                x=x,
                num_clusters=num_clusters,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
                n_init=n_init,
            )
        except Exception:
            pass
    return _run_kmeans_torch(
        x=x,
        num_clusters=num_clusters,
        max_iter=max_iter,
        tol=tol,
        restarts=restarts,
    )


def _class_prototypes(features: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    labels = torch.as_tensor(labels, device=features.device).view(-1).long()
    prototypes = features.new_zeros((num_classes, features.size(-1)))
    if features.numel() == 0:
        return prototypes

    fallback = features.mean(dim=0, keepdim=True).repeat(num_classes, 1)
    for class_id in range(num_classes):
        mask = labels == int(class_id)
        if bool(mask.any().item()):
            prototypes[class_id] = features[mask].mean(dim=0)
        else:
            prototypes[class_id] = fallback[class_id]
    return prototypes


class GPPTPrompt(nn.Module):
    """GPPT dual-token prompt head (StructureToken + per-cluster TaskToken).

    Architectural exception: unlike the other prompt modules under
    ``src/finetune/prompts`` (which are stateless bundles of learnable
    tensors), ``GPPTPrompt`` carries *training state*.  Its class centers
    are derived from a KMeans pass over the training embeddings before
    gradient-based tuning begins — this is an intrinsic part of the
    method as described in the GPPT paper and is not refactorable into
    a stateless tensor.  Prototype initialisation is driven by
    :meth:`FinetuneGPPT._maybe_initialize_prompt` so the data pass lives
    in the task class, not here; what remains here is the parameter
    storage and a one-shot ``_init_prototype_bank`` hook.  Future
    prompt modules should stay stateless unless a paper-defined
    prototype bank forces otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        center_num: int,
        num_classes: int,
        output_dim: int | None = None,
        structure_mode: str = "concat",
        task_mode: str = "concat",
        add_self_loops_in_conv: bool = False,
        kmeans_max_iter: int = 100,
        kmeans_tol: float = 1e-4,
        kmeans_restarts: int = 1,
        use_sklearn_kmeans: bool = True,
        kmeans_random_state: int = 0,
        kmeans_n_init: int = 10,
        use_prototype_init: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.center_num = max(1, int(center_num))
        self.num_classes = max(2, int(num_classes))
        self.output_dim = max(1, int(output_dim or num_classes))
        self.structure_mode = str(structure_mode or "neighbor").lower()
        self.task_mode = str(task_mode or "concat").lower()
        self.kmeans_max_iter = max(1, int(kmeans_max_iter))
        self.kmeans_tol = float(max(kmeans_tol, 0.0))
        self.kmeans_restarts = max(1, int(kmeans_restarts))
        self.use_sklearn_kmeans = bool(use_sklearn_kmeans)
        self.kmeans_random_state = int(kmeans_random_state)
        self.kmeans_n_init = max(1, int(kmeans_n_init))
        self.use_prototype_init = bool(use_prototype_init)

        valid_modes = {"node", "neighbor", "concat"}
        if self.structure_mode not in valid_modes:
            raise ValueError(f"Unsupported GPPT structure_mode: {self.structure_mode}")
        if self.task_mode not in valid_modes:
            raise ValueError(f"Unsupported GPPT task_mode: {self.task_mode}")

        self.mean_conv = SimpleMeanConv(add_self_loops_in_conv=add_self_loops_in_conv)
        structure_dim = self.in_channels * (2 if self.structure_mode == "concat" else 1)
        task_dim = self.in_channels * (2 if self.task_mode == "concat" else 1)
        self.StructureToken = nn.Linear(structure_dim, self.center_num, bias=False)
        self.TaskToken = nn.ModuleList([nn.Linear(task_dim, self.output_dim, bias=False) for _ in range(self.center_num)])
        self._mid_h: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.StructureToken.weight)
        for head in self.TaskToken:
            nn.init.xavier_uniform_(head.weight)

    @staticmethod
    def _select_feature(node_feat: Tensor, neighbor_feat: Tensor, mode: str) -> Tensor:
        if mode == "node":
            return node_feat
        if mode == "neighbor":
            return neighbor_feat
        return torch.cat([node_feat, neighbor_feat], dim=-1)

    def _build_features(self, h: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        neighbor = self.mean_conv(h, edge_index)
        structure_feat = self._select_feature(h, neighbor, self.structure_mode)
        task_feat = self._select_feature(h, neighbor, self.task_mode)
        return structure_feat, task_feat

    def get_mid_h(self) -> Tensor | None:
        return self._mid_h

    def get_TaskToken(self) -> list[Tensor]:
        return [head.weight for head in self.TaskToken]

    def get_StructureToken(self) -> Tensor:
        return self.StructureToken.weight

    def weigth_init(self, h: Tensor, edge_index: Tensor, label: Tensor, index: Tensor) -> None:
        """Initialize structure/task tokens using few-shot labels (name kept for compatibility)."""
        with torch.no_grad():
            # Run init on CPU: building full-graph structure/task features
            # spikes GPU memory on large datasets (transient multi-GB
            # allocation at row start). The init outputs (kmeans centers,
            # class prototypes) are tiny; copy_ below moves them into the
            # GPU-resident weights, and _mid_h is overwritten by the first
            # forward() before any consumer reads it.
            h = h.detach().cpu()
            edge_index = edge_index.cpu()
            if torch.is_tensor(label):
                label = label.cpu()
            if torch.is_tensor(index):
                index = index.cpu()
            structure_feat, task_feat = self._build_features(h, edge_index)
            labels = prepare_single_label_labels(label).to(structure_feat.device)

            node_index = torch.as_tensor(index, dtype=torch.long, device=structure_feat.device).view(-1)
            valid = (node_index >= 0) & (node_index < structure_feat.size(0))
            node_index = node_index[valid]
            if node_index.numel() == 0:
                node_index = torch.arange(structure_feat.size(0), device=structure_feat.device)

            selected_structure = structure_feat[node_index]
            selected_task = task_feat[node_index]
            selected_labels = labels[node_index]

            centers = _run_kmeans(
                x=selected_structure,
                num_clusters=self.center_num,
                max_iter=self.kmeans_max_iter,
                tol=self.kmeans_tol,
                restarts=self.kmeans_restarts,
                use_sklearn=self.use_sklearn_kmeans,
                random_state=self.kmeans_random_state,
                n_init=self.kmeans_n_init,
            )
            self.StructureToken.weight.data.copy_(centers)

            if self.use_prototype_init:
                prototypes = _class_prototypes(selected_task, selected_labels, self.num_classes)
                for head in self.TaskToken:
                    head.weight.data.copy_(prototypes)

            self._mid_h = structure_feat.detach()

    def update_StructureToken_weight(self, h: Tensor | None) -> None:
        if h is None:
            return
        with torch.no_grad():
            structure_feat = torch.nan_to_num(torch.as_tensor(h))
            if structure_feat.numel() == 0:
                return
            centers = _run_kmeans(
                x=structure_feat,
                num_clusters=self.center_num,
                max_iter=self.kmeans_max_iter,
                tol=self.kmeans_tol,
                restarts=self.kmeans_restarts,
                use_sklearn=self.use_sklearn_kmeans,
                random_state=self.kmeans_random_state,
                n_init=self.kmeans_n_init,
            )
            self.StructureToken.weight.data.copy_(centers)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        structure_feat, task_feat = self._build_features(h, edge_index)
        self._mid_h = structure_feat.detach()
        assignment = self.StructureToken(structure_feat).argmax(dim=1)

        out = task_feat.new_zeros((task_feat.size(0), self.output_dim))
        for center_id, head in enumerate(self.TaskToken):
            mask = assignment == center_id
            if bool(mask.any().item()):
                out[mask] = head(task_feat[mask])
        return out
