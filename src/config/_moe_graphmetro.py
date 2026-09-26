from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_graphmetro_cfg(cfg: CN) -> None:
    """Attach GraphMETRO (Wu et al., NeurIPS 2024): mixture of aligned experts under shift defaults.

    Owns the ``cfg.moe.graphmetro`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.graphmetro = CN()
    cfg.moe.graphmetro.dataset = _default_dataset_cfg()
