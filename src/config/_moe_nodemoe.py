from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_nodemoe_cfg(cfg: CN) -> None:
    """Attach Node-MoE (Han et al., 2024): node-wise filtering experts (node tasks) defaults.

    Owns the ``cfg.moe.nodemoe`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.nodemoe = CN()
    cfg.moe.nodemoe.dataset = _default_dataset_cfg()
