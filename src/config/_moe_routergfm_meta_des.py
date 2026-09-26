from yacs.config import CfgNode as CN


def set_routergfm_meta_des_cfg(cfg: CN) -> None:
    """Attach META-DES dynamic ensemble selection (Cruz et al., 2015) defaults.

    Owns ``cfg.moe.routergfm.baselines.meta_des``; attached after ``set_routergfm_cfg``.
    """
    cfg.moe.routergfm.baselines.meta_des = CN()
