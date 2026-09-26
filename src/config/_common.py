from yacs.config import CfgNode as CN


def set_general_cfg(cfg: CN) -> CN:
    """General settings shared across all workflows."""
    cfg.device = 0  # CUDA device index
    cfg.seeds = [42, 0, 100, 123, 2024]  # ordered seeds; run i uses seeds[i], single-seed workflows use seeds[0]

    return cfg
