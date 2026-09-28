from yacs.config import CfgNode as CN


def set_routergfm_kdem_ppem_cfg(cfg: CN) -> None:
    """Attach KDEM / PPEM expert merging (Liu et al., NeurIPS 2025) defaults.

    Owns ``cfg.moe.routergfm.baselines.kdem_ppem``; attached after ``set_routergfm_cfg``.
    One block serves both methods; the variant is ``baselines.method`` (kdem | ppem).
    The top-k experts of one compatible architecture group are fine-tuned on S_a
    through their merged parameters (Eq. 8) with a new task head.
    """
    b = cfg.moe.routergfm.baselines.kdem_ppem = CN()
    b.k = 3  # activated experts (paper App. G.3: best on Link2)
    b.group_policy = "top1_arch"  # top1_arch: group of the highest-competence expert | fixed: fixed_arch only
    b.fixed_arch = ""  # architecture for group_policy fixed (e.g. "gcn")
    b.competence = CN()
    b.competence.max_triplets = 10000  # sampled (anchor, positive, negative) support triplets for Eq. 5
    b.kd = CN()
    b.kd.weight = 0.01  # gamma (paper)
    b.kd.period = 10  # T1 in global steps (paper: 100 batches of one epoch; few-shot epochs have 1-7)
    b.kd.detach_teacher = True  # the paper leaves teacher gradients unspecified
    b.ema = CN()
    b.ema.period = 10  # T2 in global steps (paper grid minimum)
    # beta = target_retention ** (1 / (total_steps // period)): the paper's 0.999^6000 retention at any
    # step budget; <= 0 uses the fixed ema.beta.
    b.ema.target_retention = 0.0025
    b.ema.beta = 0.999
    b.lr_expert = 1e-4  # paper (Link1)
    b.lr_head = 1e-3
    b.weight_decay = 0.0
    b.epochs = 100
    b.early_stopping = 50  # epochs without support-loss improvement (S_a has no validation split)
    b.batch_size = 32
    b.num_workers = 0  # data loading workers (query-prediction loader)
