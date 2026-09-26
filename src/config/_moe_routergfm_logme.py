from yacs.config import CfgNode as CN


def set_routergfm_logme_cfg(cfg: CN) -> None:
    """Attach LogME transferability selection baseline (You et al., ICML 2021) defaults.

    Owns ``cfg.moe.routergfm.baselines.logme``; attached after ``set_routergfm_cfg``.
    """
    lg = cfg.moe.routergfm.baselines.logme = CN()
    lg.standardize = False  # paper/official LogME; True = legacy reranker (z-scored features and float targets)
    lg.max_iter = 100  # evidence fixed-point iterations (converged Algorithm 1, not the official 11-step early stop)
    lg.tol = 1e-6  # relative change of alpha and beta
