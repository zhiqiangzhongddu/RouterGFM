from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_linkmoe_cfg(cfg: CN) -> None:
    """Attach Link-MoE (Ma et al., NeurIPS 2024): mixture of link predictors (link tasks) defaults.

    Owns the ``cfg.moe.linkmoe`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.linkmoe = CN()
    cfg.moe.linkmoe.dataset = _default_dataset_cfg()
