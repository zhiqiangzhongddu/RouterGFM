from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_nodemoe_cfg(cfg: CN) -> None:
    """Attach Node-MoE (Han et al., 2024): node-wise filtering experts (node tasks) defaults.

    Node-MoE mixes m ChebNetII experts (diverse low/high/all-pass filter
    initialisations) with a node-wise GIN gate fed ``[x, |Ax - x|, |A^2x - x|]``
    and a filter-smoothing regulariser, trained end-to-end from scratch on the
    target support set. Node classification only; on induced ego-subgraphs the
    prediction is the target node's mixed logits.

    Owns the ``cfg.moe.nodemoe`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.nodemoe = CN()

    # ------------------------------------------------------------------ #
    # Target dataset / task (node tasks only)
    # ------------------------------------------------------------------ #
    cfg.moe.nodemoe.dataset = _default_dataset_cfg()
    cfg.moe.nodemoe.dataset.task_level = "node"  # must be node (edge/graph raise)
    cfg.moe.nodemoe.dataset.task_type = "none"  # auto-detect (classification)
    cfg.moe.nodemoe.dataset.induced = True  # ego-subgraph instances; False = original full-graph mode
    cfg.moe.nodemoe.dataset.fixed_split = (5, 0.0, 1.0)  # (shots, 0.0, 1.0) few-shot split
    cfg.moe.nodemoe.in_dim = 0  # input feature dim; inferred from dataset when 0/None

    # ------------------------------------------------------------------ #
    # Experts (ChebNetII)
    # ------------------------------------------------------------------ #
    cfg.moe.nodemoe.expert_inits = ("low", "high", "uniform")  # filter init per expert; m = len(...)
    cfg.moe.nodemoe.expert_alphas = (0.9, 0.9, 0.9)  # decay alpha per expert (same length as expert_inits)
    cfg.moe.nodemoe.K = 10  # Chebyshev polynomial order
    cfg.moe.nodemoe.expert_hidden_dim = 64  # ChebNetII MLP hidden width
    cfg.moe.nodemoe.expert_dropout = 0.5  # ChebNetII dropout
    cfg.moe.nodemoe.dprate = 0.5  # dropout before propagation
    cfg.moe.nodemoe.expert_lr = 0.01  # lr of the expert linear layers
    cfg.moe.nodemoe.expert_weight_decay = 5e-4  # weight decay of the expert linear layers
    cfg.moe.nodemoe.filter_lr = 0.01  # lr of the filter values (ChebNetII prop_lr)
    cfg.moe.nodemoe.filter_weight_decay = 5e-4  # weight decay of the filter values (ChebNetII prop_wd)
    cfg.moe.nodemoe.smoothing_gamma = 0.1  # filter-smoothing loss weight (Eq. 2)

    # ------------------------------------------------------------------ #
    # Gate (GIN)
    # ------------------------------------------------------------------ #
    cfg.moe.nodemoe.gate_hidden_dim = 64  # GIN hidden width
    cfg.moe.nodemoe.gate_num_layers = 2  # GIN layers
    cfg.moe.nodemoe.gate_dropout = 0.5  # GIN dropout
    cfg.moe.nodemoe.gate_lr = 0.01  # gate lr
    cfg.moe.nodemoe.gate_weight_decay = 5e-4  # gate weight decay
    cfg.moe.nodemoe.gate_feature_norm = "mean"  # adjacency normalisation of the gate input: mean (D^-1 A) | sym
    cfg.moe.nodemoe.readout = "target"  # induced readout: target (node's own logits) | mean (GMoE-style pooling)

    # ------------------------------------------------------------------ #
    # Training options (parity with gmoe/mowst/graphmore)
    # ------------------------------------------------------------------ #
    cfg.moe.nodemoe.num_runs = 5  # number of runs with different seeds
    cfg.moe.nodemoe.epochs = 100  # maximum number of training epochs
    cfg.moe.nodemoe.early_stopping = 50  # early stopping patience (0 disables)
    cfg.moe.nodemoe.batch_size = 32  # training/eval batch size (subgraph batches)
    cfg.moe.nodemoe.num_workers = 0  # data loading workers
    cfg.moe.nodemoe.monitor_metric = "auto"  # auto | disabled | explicit metric (see monitoring policy)
    cfg.moe.nodemoe.grad_clip = 0.0  # max gradient norm (0.0 = disabled)
    cfg.moe.nodemoe.checkpoint_dir = "outputs/moe/nodemoe/checkpoints"  # directory to save checkpoints
    cfg.moe.nodemoe.log_dir = "outputs/moe/nodemoe/logs"  # directory to save training logs
    cfg.moe.nodemoe.skip_if_exists = True  # skip run if checkpoint already exists

    # ------------------------------------------------------------------ #
    # Batch execution (TSV)
    # ------------------------------------------------------------------ #
    cfg.moe.nodemoe.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.nodemoe.tasks_tsv = "slurm/moe.nodemoe.all.tsv"  # header-based TSV of Node-MoE tasks
