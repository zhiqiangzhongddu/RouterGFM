from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_linkmoe_cfg(cfg: CN) -> None:
    """Attach Link-MoE (Ma et al., NeurIPS 2024): mixture of link predictors (link tasks) defaults.

    Two-step pipeline: heterogeneous link predictors are trained independently
    on the target's train pairs (early-stopped on val), then a gate mixing
    their probabilities from pair features (``x_i * x_j``) and structural
    heuristics is trained on the validation pairs and evaluated on test pairs.
    Expert hyperparameters are HeaRT's Cora settings (no DBLP/Cornell values
    exist); gate settings follow the official README.

    Owns the ``cfg.moe.linkmoe`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.linkmoe = CN()

    # ------------------------------------------------------------------ #
    # Target dataset / task (link prediction only)
    # ------------------------------------------------------------------ #
    cfg.moe.linkmoe.dataset = _default_dataset_cfg()
    cfg.moe.linkmoe.dataset.task_level = "edge"  # only edge (link prediction) is supported
    cfg.moe.linkmoe.dataset.task_type = "classification"
    cfg.moe.linkmoe.dataset.induced = False  # full-graph view; the SEAL expert always loads the induced view
    cfg.moe.linkmoe.dataset.fixed_split = (0.1, 0.05, 0.1)  # positive-edge train/val/test ratios

    # ------------------------------------------------------------------ #
    # Experts (step 1): mlp | gcn | ncn | seal
    # ------------------------------------------------------------------ #
    cfg.moe.linkmoe.experts = ("mlp", "gcn", "ncn", "seal")
    cfg.moe.linkmoe.expert_max_epochs = 500  # full-graph experts
    cfg.moe.linkmoe.expert_patience = 50  # epochs without val-AUC improvement (HeaRT kill_cnt 10 x eval_steps 5)
    cfg.moe.linkmoe.expert_batch_size = 1024  # train positives per batch (+ as many fresh negatives)
    cfg.moe.linkmoe.mlp = CN(dict(hidden_dim=256, num_layers=1, predictor_layers=3, dropout=0.3, lr=0.01, weight_decay=1e-4))
    cfg.moe.linkmoe.gcn = CN(dict(hidden_dim=128, num_layers=1, predictor_layers=3, dropout=0.3, lr=0.01, weight_decay=1e-4))
    cfg.moe.linkmoe.ncn = CN(dict(hidden_dim=256, num_layers=2, dropout=0.3, lr=0.01, weight_decay=1e-4, layer_norm=True, beta=1.0))
    cfg.moe.linkmoe.seal = CN(dict(
        hidden_dim=256, num_layers=3, dropout=0.5, lr=0.01, weight_decay=0.0,
        max_z=100, batch_size=256, max_epochs=100, patience=20,
    ))

    # ------------------------------------------------------------------ #
    # Gate structural heuristics (computed on the eval context graph)
    # ------------------------------------------------------------------ #
    cfg.moe.linkmoe.katz_beta = 0.005
    cfg.moe.linkmoe.ppr_damping = 0.85
    cfg.moe.linkmoe.ppr_tol = 1e-7
    cfg.moe.linkmoe.ppr_max_iter = 100

    # ------------------------------------------------------------------ #
    # Gate (step 2): trained on val pairs, selected on a held-out val slice
    # ------------------------------------------------------------------ #
    cfg.moe.linkmoe.gate = CN(dict(
        hidden_dim=64,
        num_layers=2,  # layers of each input branch (feature / structure)
        num_layers_predictor=1,  # Linear(2H -> num_experts) + softmax
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        epochs=800,  # no early stopping; keep the best gate-val AUC epoch
        val_train_ratio=0.8,  # fraction of val pairs (per class) used to train the gate
        neg_loss_weight=10.0,  # official code's negative-term weight (Eq. 3 corresponds to 1.0)
    ))

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    cfg.moe.linkmoe.num_runs = 5  # number of runs with different seeds
    cfg.moe.linkmoe.num_workers = 0  # data loading workers (SEAL loaders)
    cfg.moe.linkmoe.checkpoint_dir = "outputs/moe/linkmoe/checkpoints"
    cfg.moe.linkmoe.log_dir = "outputs/moe/linkmoe/logs"
    cfg.moe.linkmoe.skip_if_exists = True  # skip run if checkpoint already exists
    cfg.moe.linkmoe.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.linkmoe.tasks_tsv = "slurm/moe.linkmoe.all.tsv"  # header-based TSV of Link-MoE tasks
