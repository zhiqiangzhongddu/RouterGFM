from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_geomoe_cfg(cfg: CN) -> None:
    """Attach GeoMoE (Cao et al., 2026): curvature-guided geometric mixture of experts defaults.

    Owns the ``cfg.moe.geomoe`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.geomoe = CN()
    cfg.moe.geomoe.dataset = _default_dataset_cfg()
