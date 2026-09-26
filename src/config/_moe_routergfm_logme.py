from yacs.config import CfgNode as CN


def set_routergfm_logme_cfg(cfg: CN) -> None:
    """Attach LogME transferability selection baseline (You et al., ICML 2021) defaults.

    Owns ``cfg.moe.routergfm.baselines.logme``; attached after ``set_routergfm_cfg``.
    """
    cfg.moe.routergfm.baselines.logme = CN()
