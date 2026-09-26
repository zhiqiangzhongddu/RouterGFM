from yacs.config import CfgNode as CN


def set_routergfm_metadata_mlp_cfg(cfg: CN) -> None:
    """Attach metadata MLP selection baseline defaults.

    Owns ``cfg.moe.routergfm.baselines.metadata_mlp``; attached after ``set_routergfm_cfg``.
    """
    cfg.moe.routergfm.baselines.metadata_mlp = CN()
