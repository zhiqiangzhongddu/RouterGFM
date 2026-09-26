from yacs.config import CfgNode as CN


def set_routergfm_kdem_ppem_cfg(cfg: CN) -> None:
    """Attach KDEM / PPEM expert merging (Liu et al., NeurIPS 2025) defaults.

    Owns ``cfg.moe.routergfm.baselines.kdem_ppem``; attached after ``set_routergfm_cfg``.
    """
    cfg.moe.routergfm.baselines.kdem_ppem = CN()
