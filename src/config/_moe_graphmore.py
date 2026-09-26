from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_graphmore_cfg(cfg: CN) -> None:
    """Attach GraphMoRE config defaults to *cfg*.

    GraphMoRE (Guo et al., "Mitigating Topological Heterogeneity via Mixture
    of Riemannian Experts", AAAI 2025) routes each input through ``K``
    curvature-specific Riemannian (kappa-GCN) experts and combines them with
    a topology-aware gate. The reference is transductive single-graph; here,
    following the repo design, node- / edge- / graph-level tasks are all
    consumed as (induced) subgraph graph-level batches, and the gate routes
    per subgraph. Two optimizers train the model: a ``geoopt`` RiemannianAdam
    over the experts and a plain Adam over the gate + head.

    Owns the ``cfg.moe.graphmore`` subtree. Attached after the shared
    ``cfg.moe`` node via :func:`._moe.set_moe_cfg`.
    """
    cfg.moe.graphmore = CN()

    # ------------------------------------------------------------------ #
    # Target dataset / task
    # ------------------------------------------------------------------ #
    cfg.moe.graphmore.dataset = _default_dataset_cfg()
    cfg.moe.graphmore.dataset.task_level = "node"  # node, edge, or graph (induced -> graph batches)
    cfg.moe.graphmore.dataset.task_type = "none"  # classification | regression | none (auto-detect)
    cfg.moe.graphmore.dataset.induced = True  # treat node/edge tasks as induced subgraph tasks
    cfg.moe.graphmore.dataset.fixed_split = (0.8, 0.1, 0.1)  # train/val/test split ratios
    cfg.moe.graphmore.in_dim = 0  # input feature dim; inferred from dataset when 0/None

    # ------------------------------------------------------------------ #
    # Riemannian mixture-of-experts architecture
    # ------------------------------------------------------------------ #
    # One expert per initial curvature: <0 hyperbolic, >0 spherical, 0 Euclidean
    # (Euclidean experts keep a fixed curvature; signed ones are learnable).
    cfg.moe.graphmore.init_curvs = [-1.0, 0.0, 1.0]  # K = len(init_curvs) experts
    cfg.moe.graphmore.hidden_dim = 64  # kappa-GCN hidden dim (experts are 2-layer)
    cfg.moe.graphmore.embed_dim = 32  # per-expert output dim; mixture dim = K * embed_dim
    cfg.moe.graphmore.learnable = True  # learn signed curvatures during training
    cfg.moe.graphmore.gating_hidden_dim = 32  # hidden dim of the per-subgraph gating GCN
    cfg.moe.graphmore.gating_temperature = 1.0  # softmax temperature for the gate
    cfg.moe.graphmore.graph_pooling = "mean"  # mean, max, or sum pooling for graph readout
    # Distortion regulariser coefficient. Off by default: on small induced
    # subgraphs the paper's whole-graph distortion term degenerates (intra-
    # subgraph edges are ~all distance-1).
    cfg.moe.graphmore.coef_dis = 0.0

    # ------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------ #
    cfg.moe.graphmore.num_runs = 5  # number of runs with different seeds
    cfg.moe.graphmore.epochs = 100  # maximum number of training epochs
    cfg.moe.graphmore.early_stopping = 50  # early stopping patience (0 disables)
    cfg.moe.graphmore.lr = 1e-2  # Adam learning rate (gating + head)
    cfg.moe.graphmore.weight_decay = 5e-4  # Adam weight decay (gating + head)
    cfg.moe.graphmore.lr_riemann = 1e-2  # RiemannianAdam learning rate (experts)
    cfg.moe.graphmore.weight_decay_riemann = 0.0  # RiemannianAdam weight decay (experts)
    cfg.moe.graphmore.batch_size = 32  # training/eval batch size (graph batches)
    cfg.moe.graphmore.num_workers = 0  # data loading workers
    cfg.moe.graphmore.monitor_metric = "auto"  # auto | disabled | explicit metric (see monitoring policy)
    cfg.moe.graphmore.scheduler = "none"  # learning rate scheduler: none, cosine, step
    cfg.moe.graphmore.scheduler_step_size = 50  # step size for StepLR scheduler
    cfg.moe.graphmore.scheduler_gamma = 0.5  # decay factor for StepLR scheduler
    cfg.moe.graphmore.grad_clip = 0.0  # max gradient norm (0.0 = disabled)
    cfg.moe.graphmore.checkpoint_dir = "outputs/moe/graphmore/checkpoints"  # directory to save checkpoints
    cfg.moe.graphmore.log_dir = "outputs/moe/graphmore/logs"  # directory to save training logs
    cfg.moe.graphmore.skip_if_exists = True  # skip run if checkpoint already exists

    # ------------------------------------------------------------------ #
    # Batch execution (TSV)
    # ------------------------------------------------------------------ #
    cfg.moe.graphmore.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.graphmore.tasks_tsv = "slurm/moe.graphmore.all.tsv"  # header-based TSV of GraphMoRE tasks
