"""MetaGL (Park et al., ICLR 2023) and MetaGL+metadata selection baselines.

``MetaGLSelector(cfg, infra)`` is registered as ``metagl``; with
``use_metadata=True`` as ``metagl_metadata`` (expert-insertion extension).
Both read ``cfg.moe.routergfm.baselines.metagl``.
"""

from .selector import MetaGLSelector

__all__ = ["MetaGLSelector"]
