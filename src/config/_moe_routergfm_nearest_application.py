from yacs.config import CfgNode as CN


def set_routergfm_nearest_application_cfg(cfg: CN) -> None:
    """Attach nearest-application selection baseline defaults.

    Owns ``cfg.moe.routergfm.baselines.nearest_application``; attached after ``set_routergfm_cfg``.
    """
    cfg.moe.routergfm.baselines.nearest_application = CN()
