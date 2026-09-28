from yacs.config import CfgNode as CN


def set_routergfm_meta_des_cfg(cfg: CN) -> None:
    """Attach META-DES dynamic ensemble selection (Cruz et al., 2015) defaults.

    Owns ``cfg.moe.routergfm.baselines.meta_des``; attached after ``set_routergfm_cfg``.
    The pool comes from ``baselines.candidate_rule`` / ``candidate_pool``; out-of-fold
    support posteriors come from the infra heads (``heads.oof_folds``).
    """
    m = cfg.moe.routergfm.baselines.meta_des = CN()
    m.k = 7  # region of competence size in descriptor space (paper Sec. 4.3)
    m.kp = 5  # output-profile neighbours (paper Sec. 4.3)
    m.hc = 0.7  # consensus threshold of the meta-training sample selection (paper)
    m.selection_threshold = 0.5  # competent iff lambda(v) > threshold (DESlib)
    m.meta_hidden = 10  # hidden units of the meta-classifier lambda (paper)
    m.meta_val_frac = 0.25  # validation share of the meta-training set (paper 75/25)
    m.meta_patience = 5  # epochs without validation improvement before stopping (paper)
    m.meta_max_epochs = 200  # L-BFGS epochs cap
    m.query_chunk_size = 16384  # queries per chunk of the streaming generalization phase
