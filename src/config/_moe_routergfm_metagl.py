from yacs.config import CfgNode as CN


def set_routergfm_metagl_cfg(cfg: CN) -> None:
    """Attach MetaGL / MetaGL+metadata selection baseline (Park et al., ICLR 2023) defaults.

    Owns ``cfg.moe.routergfm.baselines.metagl``; attached after ``set_routergfm_cfg``.
    Values follow the official code (paper App. E.1); also used by MetaGL-U.
    """
    m = cfg.moe.routergfm.baselines.metagl = CN()
    m.hid_dim = 32  # HGT embedding size; latent factors k_in = min(2 * hid_dim, meta dim, #experts)
    m.knn_k = 30  # neighbours per node in every kNN relation of the G-M network
    m.hgt_layers = 2
    m.hgt_heads = 4
    m.hgt_dropout = 0.5
    m.lr = 7.5e-4
    m.weight_decay = 1e-4
    m.epochs = 500
    m.patience = 50  # epochs without validation improvement (criterion mean(2 AUC, MRR))
    m.batch_size = 80  # training applications per step
    m.val_ratio = 0.3  # share of base datasets held out for early stopping
    m.rf_n_estimators = 100  # random forests M' -> U (and metadata -> V for MetaGL+metadata)
    m.min_slice_rows = 10  # smaller (task family, budget) slices pool all same-budget applications
    m.graph_sample_max = 1000  # graph level only: instance graphs in the disjoint union behind the meta-graph features
    m.graph_sample_seed = 0
