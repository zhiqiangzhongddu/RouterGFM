from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_gmoe_cfg(cfg: CN) -> None:
    """Attach GMoE config defaults to *cfg*.

    GMoE (Graph Mixture of Experts, NeurIPS 2023) replaces each GNN
    message-passing layer with a sparsely-gated mixture of GNN-conv
    experts that span different receptive fields (1-hop vs 2-hop). A
    noisy top-k gate routes every node to ``k`` experts, and a
    load-balancing loss keeps expert usage uniform. GMoE is trained
    end-to-end on a single dataset; following the repo design, node- /
    edge- / graph-level tasks are all consumed as (induced) subgraph
    graph-level batches.

    Owns the ``cfg.moe.gmoe`` subtree. Attached after the shared
    ``cfg.moe`` node via :func:`._moe.set_moe_cfg`.
    """
    cfg.moe.gmoe = CN()

    # ------------------------------------------------------------------ #
    # Target dataset / task
    # ------------------------------------------------------------------ #
    cfg.moe.gmoe.dataset = _default_dataset_cfg()
    cfg.moe.gmoe.dataset.task_level = "node"  # node, edge, or graph (induced -> graph batches)
    # "none" lets populate_dataset_cfg_from_meta auto-detect classification vs
    # regression from the dataset (matching the train workflow). A non-sentinel
    # default would block auto-fill and silently train regression datasets as
    # classification; override explicitly (e.g. "regression") only when needed.
    cfg.moe.gmoe.dataset.task_type = "none"  # classification | regression | none (auto-detect)
    cfg.moe.gmoe.dataset.induced = True  # treat node/edge tasks as induced subgraph tasks
    cfg.moe.gmoe.dataset.fixed_split = (0.8, 0.1, 0.1)  # train/val/test split ratios
    cfg.moe.gmoe.in_dim = 0  # input feature dim; inferred from dataset when 0/None

    # ------------------------------------------------------------------ #
    # GMoE architecture
    # ------------------------------------------------------------------ #
    cfg.moe.gmoe.gnn_type = "gcn"  # expert conv type: gcn or gin
    cfg.moe.gmoe.num_layers = 5  # number of MoE message-passing layers (must be >= 2)
    cfg.moe.gmoe.hidden_dim = 128  # hidden / output embedding dim (constant across layers)
    cfg.moe.gmoe.dropout = 0.5  # dropout rate applied after each layer
    cfg.moe.gmoe.residual = False  # add residual connection between layers
    cfg.moe.gmoe.jk = "last"  # jumping-knowledge readout: last or sum
    cfg.moe.gmoe.graph_pooling = "mean"  # mean, max, or sum pooling for graph readout
    cfg.moe.gmoe.use_batchnorm = True  # per-expert BatchNorm1d (GMoE reference uses BN)

    # ------------------------------------------------------------------ #
    # Mixture-of-experts options
    # ------------------------------------------------------------------ #
    cfg.moe.gmoe.num_experts = 8  # total experts per layer
    cfg.moe.gmoe.num_experts_1hop = 4  # first n1 experts are 1-hop; the rest are multi-hop
    cfg.moe.gmoe.k = 4  # top-k experts selected per node by the gate
    cfg.moe.gmoe.coef = 1.0  # load-balancing loss coefficient (added to task loss)
    cfg.moe.gmoe.noisy_gating = True  # add tunable Gaussian noise to gate logits at train time
    cfg.moe.gmoe.expert_hop = 2  # receptive field (in hops) of the non-1hop experts

    # ------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------ #
    cfg.moe.gmoe.num_runs = 5  # number of runs with different seeds
    cfg.moe.gmoe.epochs = 100  # maximum number of training epochs
    cfg.moe.gmoe.early_stopping = 50  # early stopping patience (0 disables)
    cfg.moe.gmoe.lr = 1e-3  # learning rate
    cfg.moe.gmoe.weight_decay = 0.0  # weight decay
    cfg.moe.gmoe.batch_size = 32  # training/eval batch size (graph batches)
    cfg.moe.gmoe.num_workers = 0  # data loading workers
    cfg.moe.gmoe.monitor_metric = "auto"  # auto | disabled | explicit metric (see monitoring policy)
    cfg.moe.gmoe.scheduler = "none"  # learning rate scheduler: none, cosine, step
    cfg.moe.gmoe.scheduler_step_size = 50  # step size for StepLR scheduler
    cfg.moe.gmoe.scheduler_gamma = 0.5  # decay factor for StepLR scheduler
    cfg.moe.gmoe.grad_clip = 0.0  # max gradient norm (0.0 = disabled)
    cfg.moe.gmoe.checkpoint_dir = "outputs/moe/gmoe/checkpoints"  # directory to save checkpoints
    cfg.moe.gmoe.log_dir = "outputs/moe/gmoe/logs"  # directory to save training logs
    cfg.moe.gmoe.skip_if_exists = True  # skip run if checkpoint already exists

    # ------------------------------------------------------------------ #
    # Batch execution (TSV)
    # ------------------------------------------------------------------ #
    cfg.moe.gmoe.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.gmoe.tasks_tsv = "slurm/moe.gmoe.all.tsv"  # header-based TSV of GMoE tasks
