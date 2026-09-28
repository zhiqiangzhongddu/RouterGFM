from yacs.config import CfgNode as CN


def set_routergfm_metagl_u_cfg(cfg: CN) -> None:
    """Attach MetaGL-U: uniform mixture of the MetaGL-selected team defaults.

    Owns ``cfg.moe.routergfm.baselines.metagl_u``; attached after ``set_routergfm_cfg``.
    The team size is ``baselines.topk``; selector hyperparameters are ``baselines.metagl``.
    """
    u = cfg.moe.routergfm.baselines.metagl_u = CN()
    u.selector = "metagl"  # metagl | metagl_metadata
