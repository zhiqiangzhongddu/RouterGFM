from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_mowst_cfg(cfg: CN) -> None:
    """Attach Mowst (Mixture of Weak and Strong experts) config defaults.

    Mowst (Zeng et al., ICLR 2024) mixes, per sample, a *weak* expert (an MLP
    over features) and a *strong* expert (a message-passing GNN) through a
    learned gate that routes each sample by the *dispersion* (variance +
    entropy) of the weak expert's prediction. Two variants are provided:
    ``mowst_star`` (joint gating — one optimizer over both experts and the
    gate) and ``mowst`` (alternating weak/strong turn training). Like
    GMoE/GraphMoRE it is a single model trained end-to-end on one dataset;
    following the repo design, node- / edge- / graph-level tasks are all
    consumed as (induced) subgraph graph-level batches.

    Owns the ``cfg.moe.mowst`` subtree. Attached after the shared ``cfg.moe``
    node via :func:`._moe.set_moe_cfg`.
    """
    cfg.moe.mowst = CN()

    # ------------------------------------------------------------------ #
    # Method selection
    # ------------------------------------------------------------------ #
    cfg.moe.mowst.variant = "mowst_star"  # mowst_star (joint gating) | mowst (alternating turns)
    cfg.moe.mowst.subloss = "joint"  # joint (mix logits) | separate (per-sample gate-weighted); mowst forces separate
    # submethod — warm-up the experts before the gated stage:
    #   none | pretrain_model1 (weak) | pretrain_model2 (strong) | pretrain_both
    cfg.moe.mowst.submethod = "pretrain_model2"
    cfg.moe.mowst.pretrain_epochs = 100  # per-expert warm-up epochs (submethod != none)
    # original_data — when True the gate also sees the weak expert's pooled
    # embedding (analogue of the reference concatenating raw node features).
    cfg.moe.mowst.original_data = False

    # ------------------------------------------------------------------ #
    # Target dataset / task (all task levels -> induced subgraph batches)
    # ------------------------------------------------------------------ #
    cfg.moe.mowst.dataset = _default_dataset_cfg()
    cfg.moe.mowst.dataset.task_level = "node"  # node, edge, or graph (induced -> graph batches)
    # "none" lets populate_dataset_cfg_from_meta auto-detect classification vs
    # regression from the dataset (matching the train workflow).
    cfg.moe.mowst.dataset.task_type = "none"  # classification | regression | none (auto-detect)
    cfg.moe.mowst.dataset.induced = True  # treat node/edge tasks as induced subgraph tasks
    cfg.moe.mowst.dataset.fixed_split = (0.8, 0.1, 0.1)  # train/val/test split ratios
    cfg.moe.mowst.in_dim = 0  # input feature dim; inferred from dataset when 0/None

    # ------------------------------------------------------------------ #
    # Experts and gate
    # ------------------------------------------------------------------ #
    # Both experts output a ``hidden_dim`` embedding; the per-expert supervised
    # heads (built in the task) map that to logits.
    cfg.moe.mowst.hidden_dim = 128  # expert embedding dim (head input) + internal hidden width

    cfg.moe.mowst.weak = CN()
    cfg.moe.mowst.weak.model = "mlp"  # weak expert backbone (typically mlp)
    cfg.moe.mowst.weak.num_layers = 2
    cfg.moe.mowst.weak.dropout = 0.5

    cfg.moe.mowst.strong = CN()
    cfg.moe.mowst.strong.model = "gcn"  # gcn | gin | gat | mlp
    cfg.moe.mowst.strong.num_layers = 2
    cfg.moe.mowst.strong.dropout = 0.5
    cfg.moe.mowst.strong.gat_heads = 2  # only used when strong.model == gat

    cfg.moe.mowst.gate = CN()
    cfg.moe.mowst.gate.hidden_dim = 64
    cfg.moe.mowst.gate.num_layers = 2
    cfg.moe.mowst.gate.dropout = 0.5

    cfg.moe.mowst.activation = "relu"
    cfg.moe.mowst.use_batchnorm = True  # BatchNorm in experts/gate (reference uses BN)
    cfg.moe.mowst.graph_pooling = "mean"  # mean | max | sum readout for graph-level batches

    # ------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------ #
    cfg.moe.mowst.num_runs = 5  # number of runs with different seeds
    cfg.moe.mowst.epochs = 100  # maximum number of training epochs
    cfg.moe.mowst.early_stopping = 50  # early stopping patience (0 disables)
    cfg.moe.mowst.lr = 1e-3  # LR for per-expert pretraining + the mowst alternating turns
    cfg.moe.mowst.lr_gate = 1e-3  # LR for the mowst_star joint (experts + gate) optimizer
    cfg.moe.mowst.weight_decay = 0.0  # weight decay
    cfg.moe.mowst.batch_size = 32  # training/eval batch size (graph batches)
    cfg.moe.mowst.num_workers = 0  # data loading workers
    cfg.moe.mowst.monitor_metric = "auto"  # auto | disabled | explicit metric (see monitoring policy)
    cfg.moe.mowst.scheduler = "none"  # learning rate scheduler: none, cosine, step
    cfg.moe.mowst.scheduler_step_size = 50  # step size for StepLR scheduler
    cfg.moe.mowst.scheduler_gamma = 0.5  # decay factor for StepLR scheduler
    cfg.moe.mowst.grad_clip = 0.0  # max gradient norm (0.0 = disabled)
    cfg.moe.mowst.checkpoint_dir = "outputs/moe/mowst/checkpoints"  # directory to save checkpoints
    cfg.moe.mowst.log_dir = "outputs/moe/mowst/logs"  # directory to save training logs
    cfg.moe.mowst.skip_if_exists = True  # skip run if checkpoint already exists

    # ------------------------------------------------------------------ #
    # Batch execution (TSV)
    # ------------------------------------------------------------------ #
    cfg.moe.mowst.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.mowst.tasks_tsv = "slurm/moe.mowst.all.tsv"  # header-based TSV of Mowst tasks
