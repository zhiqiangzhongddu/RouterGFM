from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_gmope_cfg(cfg: CN) -> None:
    """Attach GMoPE (Wang et al., 2025): graph mixture of prompt-experts defaults.

    GMoPE pretrains M prompt-conditioned GCN experts (each expert owns a prompt
    concatenated to every node feature) on the source corpora with a
    structure-aware soft top-K router and a soft orthogonality loss on the
    prompts, then freezes the experts and tunes only the prompts and a shared
    task head on the target support set (hard top-K router). Inference mixes
    the experts' pooled embeddings with normalised-entropy confidence weights.
    Pretraining runs once per route (``node`` serves node/link targets,
    ``graph`` serves graph targets) and is reused by every target, budget and
    seed. Node/edge targets are consumed as induced-subgraph graph batches.

    Owns the ``cfg.moe.gmope`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.gmope = CN()
    cfg.moe.gmope.stage = "all"  # all (pretrain route if missing, then finetune) | pretrain | finetune

    # ------------------------------------------------------------------ #
    # Target dataset / task
    # ------------------------------------------------------------------ #
    cfg.moe.gmope.dataset = _default_dataset_cfg()
    cfg.moe.gmope.dataset.task_level = "node"  # node, edge, or graph (induced -> graph batches)
    cfg.moe.gmope.dataset.task_type = "none"  # classification | regression | none (auto-detect)
    cfg.moe.gmope.dataset.induced = True  # treat node/edge tasks as induced subgraph tasks
    cfg.moe.gmope.dataset.fixed_split = (5, 0.0, 1.0)  # few-shot support budget (shots, 0.0, 1.0) or LP ratios
    cfg.moe.gmope.in_dim = 100  # aligned feature dim d0; asserted equal to every source/target feature dim

    # ------------------------------------------------------------------ #
    # Experts and prompts
    # ------------------------------------------------------------------ #
    cfg.moe.gmope.expert = CN()
    cfg.moe.gmope.expert.gnn_type = "gcn"  # expert backbone (paper: GCN)
    cfg.moe.gmope.expert.num_layers = 3  # paper complexity analysis assumes 3-layer GCN experts
    cfg.moe.gmope.expert.hidden_dim = 128  # hidden dim (paper implies 64; 128 matches the repo encoders)
    cfg.moe.gmope.expert.out_dim = 128  # output embedding dim
    cfg.moe.gmope.expert.dropout = 0.5  # dropout between expert layers
    cfg.moe.gmope.expert.graph_pooling = "mean"  # readout of the (sub)graph embedding
    cfg.moe.gmope.expert.use_batchnorm = False  # BatchNorm1d after hidden layers
    cfg.moe.gmope.num_experts = 0  # M; 0 -> number of pretraining sources of the route (paper: M = N)
    cfg.moe.gmope.prompt_dim = 32  # d_p; paper range [d0/4, d0/2]
    cfg.moe.gmope.tau = 0.8  # soft-router temperature (paper)
    cfg.moe.gmope.ortho_weight = 1.0  # lambda on the soft orthogonality loss (paper range (0, 3))

    # ------------------------------------------------------------------ #
    # Stage A: multi-source pretraining (one checkpoint per route)
    # ------------------------------------------------------------------ #
    cfg.moe.gmope.pretrain = CN()
    cfg.moe.gmope.pretrain.routes = ["node", "graph"]  # routes pretrained by stage=pretrain
    cfg.moe.gmope.pretrain.node_sources = ["cora", "pubmed", "computers", "flickr", "actor", "squirrel", "email"]
    cfg.moe.gmope.pretrain.graph_sources = ["bbbp", "tox21", "qm9", "proteins", "cifar10"]
    cfg.moe.gmope.pretrain.objective = "edge_pred"  # registered PretrainTask; reads cfg.pretrain.<objective>.*
    cfg.moe.gmope.pretrain.top_k = 0  # K of the soft router; 0 -> node route: M, graph route: 1
    cfg.moe.gmope.pretrain.epochs = 200  # paper: 150-200
    cfg.moe.gmope.pretrain.lr = 1e-3  # learning rate (experts, prompts, objective heads)
    cfg.moe.gmope.pretrain.weight_decay = 0.0  # weight decay
    cfg.moe.gmope.pretrain.batch_size = 128  # graphs per (homogeneous, single-source) batch
    cfg.moe.gmope.pretrain.max_batches_per_source = 100  # per-epoch batch cap per source (<=0 disables)
    cfg.moe.gmope.pretrain.num_workers = 0  # data loading workers
    cfg.moe.gmope.pretrain.seed_policy = "shared"  # shared (cfg.seeds[0] for every run) | per_seed (cfg.seed)
    cfg.moe.gmope.pretrain.checkpoint_dir = "outputs/moe/gmope/pretrained"  # <dir>/<route>/<run_name>.pt
    cfg.moe.gmope.pretrain.skip_if_exists = True  # reuse an existing route checkpoint

    # ------------------------------------------------------------------ #
    # Stage B: prompt tuning on the target support set (experts frozen)
    # ------------------------------------------------------------------ #
    cfg.moe.gmope.finetune = CN()
    cfg.moe.gmope.finetune.top_k = 0  # K of the hard router; 0 -> node/edge targets: M, graph targets: 1
    cfg.moe.gmope.finetune.route_loss = "pretrain"  # routing score: pretrain (label-free objective) | task (support loss)
    cfg.moe.gmope.finetune.epochs = 200  # maximum number of prompt-tuning epochs
    cfg.moe.gmope.finetune.early_stopping = 50  # early stopping patience (0 disables)
    cfg.moe.gmope.finetune.lr = 1e-3  # learning rate (prompts and head)
    cfg.moe.gmope.finetune.weight_decay = 0.0  # weight decay
    cfg.moe.gmope.finetune.batch_size = 32  # training/eval batch size (graph batches)
    cfg.moe.gmope.finetune.num_workers = 0  # data loading workers
    cfg.moe.gmope.finetune.monitor_metric = "auto"  # auto | disabled | explicit metric (see monitoring policy)
    cfg.moe.gmope.finetune.grad_clip = 0.0  # max gradient norm (0.0 = disabled)

    # ------------------------------------------------------------------ #
    # Stage C: confidence-guided aggregation
    # ------------------------------------------------------------------ #
    cfg.moe.gmope.aggregation = CN()
    cfg.moe.gmope.aggregation.experts = "all"  # all (paper Eq. 13-14) | routed (top-K of the label-free route score)

    # ------------------------------------------------------------------ #
    # Runs, outputs, batch execution
    # ------------------------------------------------------------------ #
    cfg.moe.gmope.num_runs = 5  # number of runs with different seeds
    cfg.moe.gmope.checkpoint_dir = "outputs/moe/gmope/checkpoints"  # directory to save finetune checkpoints
    cfg.moe.gmope.log_dir = "outputs/moe/gmope/logs"  # directory to save training logs
    cfg.moe.gmope.skip_if_exists = True  # skip run if checkpoint already exists
    cfg.moe.gmope.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.gmope.tasks_tsv = "slurm/moe.gmope.all.tsv"  # header-based TSV of GMoPE tasks
