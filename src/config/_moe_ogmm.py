from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_ogmm_cfg(cfg: CN) -> None:
    """Attach OGMM (Wang et al., ICLR 2026): out-of-distribution graph models merging defaults.

    Owns the ``cfg.moe.ogmm`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.ogmm = CN()
    cfg.moe.ogmm.dataset = _default_dataset_cfg()
