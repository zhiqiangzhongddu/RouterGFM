from yacs.config import CfgNode as CN


def set_routergfm_metadata_mlp_cfg(cfg: CN) -> None:
    """Attach metadata MLP selection baseline defaults.

    Owns ``cfg.moe.routergfm.baselines.metadata_mlp``; attached after ``set_routergfm_cfg``.
    The paper specifies no hyperparameters; width and optimizer mirror the router scorer.
    """
    m = cfg.moe.routergfm.baselines.metadata_mlp = CN()
    m.hidden_dim = 128
    m.num_layers = 2  # hidden layers
    m.dropout = 0.1
    m.lr = 3e-4
    m.weight_decay = 1e-4
    m.epochs = 500
    m.patience = 50  # epochs without validation regret@K improvement
    m.huber_delta = 1.0
    m.val_group_frac = 0.2  # share of historical base datasets held out for early stopping
