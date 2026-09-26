from yacs.config import CfgNode as CN


def set_routergfm_metagl_cfg(cfg: CN) -> None:
    """Attach MetaGL / MetaGL+metadata selection baseline (Park et al., ICLR 2023) defaults.

    Owns ``cfg.moe.routergfm.baselines.metagl``; attached after ``set_routergfm_cfg``.
    """
    cfg.moe.routergfm.baselines.metagl = CN()
