from yacs.config import CfgNode as CN


def set_routergfm_model_spider_cfg(cfg: CN) -> None:
    """Attach Model Spider selection baseline (Zhang et al., NeurIPS 2023) defaults.

    Owns ``cfg.moe.routergfm.baselines.model_spider``; attached after ``set_routergfm_cfg``.
    Training/validation groups follow ``router.val_datasets`` / ``router.num_val_datasets``.
    """
    ms = cfg.moe.routergfm.baselines.model_spider = CN()
    ms.token_dim = 128  # d (paper 1024): width of the expert readouts, 588 per-expert projections
    ms.num_heads = 1
    ms.dropout = 0.1
    ms.type_prompts = False  # paper: True; official reproduction script: False
    ms.specific_token_mode = "append"  # append [theta; G; S] (official code) | replace [theta; S] (paper)
    ms.epochs = 30
    ms.batch_size = 16  # historical applications per update
    ms.lr = 2.5e-4
    ms.weight_decay = 0.0  # official Adam ignores the script's 5e-4 flag
    ms.lr_min = 5e-6  # cosine eta_min, stepped once per epoch
    ms.train_specific_min = 0  # k ~ U{min..max} experts per task get specific tokens during training
    ms.train_specific_max = 10
    ms.rerank_topk_grid = [0, 3, 5, 10]  # k_r (target support executions), chosen on validation applications
    ms.regression_bins = 5  # support quantile bins of the first regression target (token partition)
    ms.checkpoint_dir = ""  # '' -> <output_root>/model_spider/checkpoints
    ms.cache_dir = ""  # '' -> <output_root>/model_spider/cache (per-expert support class centres)
