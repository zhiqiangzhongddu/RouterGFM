from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_gmope_cfg(cfg: CN) -> None:
    """Attach GMoPE (Wang et al., 2025): graph mixture of prompt-experts defaults.

    Owns the ``cfg.moe.gmope`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.gmope = CN()
    cfg.moe.gmope.dataset = _default_dataset_cfg()
