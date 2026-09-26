from yacs.config import CfgNode as CN


def set_routergfm_sagmm_pe_cfg(cfg: CN) -> None:
    """Attach SAGMM-PE: self-adaptive graph mixture over frozen pretrained experts (Meena et al., AAAI 2026) defaults.

    Owns ``cfg.moe.routergfm.baselines.sagmm_pe``; attached after ``set_routergfm_cfg``.
    """
    cfg.moe.routergfm.baselines.sagmm_pe = CN()
