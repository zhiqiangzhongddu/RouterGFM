from yacs.config import CfgNode as CN


def set_routergfm_model_spider_cfg(cfg: CN) -> None:
    """Attach Model Spider selection baseline (Zhang et al., NeurIPS 2023) defaults.

    Owns ``cfg.moe.routergfm.baselines.model_spider``; attached after ``set_routergfm_cfg``.
    """
    cfg.moe.routergfm.baselines.model_spider = CN()
