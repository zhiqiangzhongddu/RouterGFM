from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_geomoe_cfg(cfg: CN) -> None:
    """Attach GeoMoE (Cao et al., 2026): curvature-guided geometric mixture of experts defaults.

    Euclidean / hyperbolic / spherical GNN experts fused per node by a
    graph-aware gate, trained end-to-end on the target support with the task
    loss plus Ollivier-Ricci-curvature alignment and contrastive terms. A
    shift-oriented baseline of RouterGFM Table 15 (own expert inventory); the
    shift condition is selected with ``data_preparation.dataset.split_root``.
    Values marked PROPOSED are not given by the paper (no official code).

    Owns the ``cfg.moe.geomoe`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.geomoe = CN()

    # ------------------------------------------------------------------ #
    # Target dataset / task
    # ------------------------------------------------------------------ #
    cfg.moe.geomoe.dataset = _default_dataset_cfg()
    cfg.moe.geomoe.dataset.task_level = "node"  # node, edge, or graph (node/edge need induced subgraphs)
    cfg.moe.geomoe.dataset.task_type = "none"  # classification | regression | none (auto-detect)
    cfg.moe.geomoe.dataset.induced = True  # node/edge tasks as ego / enclosing subgraph instances
    cfg.moe.geomoe.dataset.fixed_split = (5, 0.0, 1.0)  # few-shot support budget (shots, 0.0, 1.0)
    cfg.moe.geomoe.in_dim = 0  # input feature dim; inferred from dataset when 0/None

    # ------------------------------------------------------------------ #
    # Geometric experts and gate
    # ------------------------------------------------------------------ #
    cfg.moe.geomoe.hidden_dim = 16  # latent dim of every expert (paper: 16 for all methods)
    cfg.moe.geomoe.num_layers = 2  # Euclidean GCN layers; kappa-GCN experts are 2-layer (PROPOSED)
    cfg.moe.geomoe.dropout = 0.5  # Euclidean GCN and gate MLP dropout (PROPOSED, paper grid 0.0-0.6)
    cfg.moe.geomoe.curvatures = [-1.0, 1.0]  # fixed (hyperbolic, spherical) curvatures (PROPOSED)
    cfg.moe.geomoe.gate_temperature = 1.0  # tau_g of the gate softmax (PROPOSED)
    cfg.moe.geomoe.graph_pooling = "mean"  # graph-level readout; node/edge read the target node / endpoints

    # ------------------------------------------------------------------ #
    # Curvature-guided objective: alpha*L_task + beta*L_align + gamma*L_contr
    # ------------------------------------------------------------------ #
    cfg.moe.geomoe.theta = 1e-4  # ORC region threshold (paper best, Fig. 4)
    cfg.moe.geomoe.eta = 0.1  # smoothing of the ORC target weights (PROPOSED)
    cfg.moe.geomoe.orc_idleness = 0.5  # lazy random-walk idleness p (PROPOSED)
    cfg.moe.geomoe.num_negatives = 4  # K: 2 intra-node + K-2 inter-node hard negatives (paper best on Photo)
    cfg.moe.geomoe.contrast_temperature = 0.5  # tau_c (PROPOSED)
    cfg.moe.geomoe.alpha = 1.0  # task-loss weight (PROPOSED)
    cfg.moe.geomoe.beta = 1.0  # alignment-loss weight (PROPOSED)
    cfg.moe.geomoe.gamma = 1.0  # contrastive-loss weight (PROPOSED)

    # ------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------ #
    cfg.moe.geomoe.num_runs = 5  # number of runs with different seeds
    cfg.moe.geomoe.epochs = 100  # maximum number of training epochs (PROPOSED)
    cfg.moe.geomoe.early_stopping = 0  # early stopping patience (0 disables)
    cfg.moe.geomoe.lr = 1e-2  # Adam learning rate (paper grid {1e-3, 5e-3, 1e-2, 2e-2})
    cfg.moe.geomoe.weight_decay = 5e-4  # Adam weight decay (paper grid {0, 1e-4, 5e-4, 1e-3})
    cfg.moe.geomoe.batch_size = 32  # instances per batch (also the hard-negative pool)
    cfg.moe.geomoe.num_workers = 0  # data loading workers
    cfg.moe.geomoe.monitor_metric = "auto"  # auto | disabled | explicit metric (few-shot auto -> train_loss)
    cfg.moe.geomoe.grad_clip = 0.0  # max gradient norm (0.0 = disabled)
    cfg.moe.geomoe.orc_cache_dir = "outputs/moe/geomoe/orc_cache"  # support ORC cache, content-addressed ("" disables)
    cfg.moe.geomoe.checkpoint_dir = "outputs/moe/geomoe/checkpoints"  # directory to save checkpoints
    cfg.moe.geomoe.log_dir = "outputs/moe/geomoe/logs"  # directory to save training logs
    cfg.moe.geomoe.prediction_dir = "outputs/moe/geomoe/predictions"  # query predictions for the Brier risk
    cfg.moe.geomoe.skip_if_exists = True  # skip run if checkpoint already exists

    # ------------------------------------------------------------------ #
    # Batch execution (TSV)
    # ------------------------------------------------------------------ #
    cfg.moe.geomoe.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.geomoe.tasks_tsv = "slurm/moe.geomoe.all.tsv"  # header-based TSV of GeoMoE tasks
