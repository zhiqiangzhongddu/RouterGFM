from yacs.config import CfgNode as CN


def set_routergfm_nearest_application_cfg(cfg: CN) -> None:
    """Attach nearest-application selection baseline defaults.

    Owns ``cfg.moe.routergfm.baselines.nearest_application``; attached after ``set_routergfm_cfg``.
    """
    n = cfg.moe.routergfm.baselines.nearest_application = CN()
    n.restrict_compatible = True  # candidates share the target's task family and budget (budget ignored for LP)
    n.tie_tol = 1e-9  # cosine-similarity tolerance of the nearest (tie) group
