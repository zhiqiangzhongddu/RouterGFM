from yacs.config import CfgNode as CN


def set_routergfm_sagmm_pe_cfg(cfg: CN) -> None:
    """Attach SAGMM-PE: self-adaptive graph mixture over frozen pretrained experts (Meena et al., AAAI 2026) defaults.

    Owns ``cfg.moe.routergfm.baselines.sagmm_pe``; attached after ``set_routergfm_cfg``.
    Base keys follow the official node script (run_arxiv.sh); the ``edge`` and ``graph``
    sub-blocks override them for link and graph-level applications (run_collab.sh,
    graph_run.sh). Experts come from ``baselines.candidate_rule`` / ``candidate_pool``.
    """
    s = cfg.moe.routergfm.baselines.sagmm_pe = CN()
    s.expert_norm = "l2"  # l2 | none: per-expert L2 normalisation of frozen readouts (--add_expert_norm)
    # Laplacian eigenvector columns of the gate input; 0 under the induced protocol, because per-instance
    # eigenvectors and signs have no cross-instance meaning; -1 -> feature dim (official full-graph X_g)
    s.gate_input_p = 0
    s.score_act = "sigmoid"  # sigmoid | softplus
    s.threshold_init = "zeros"  # zeros | randn (0.1 * N(0, 1))
    s.imp_weight = 0.1  # importance (cv^2) auxiliary loss
    s.div_weight = 0.05  # gate diversity auxiliary loss
    s.prune = True
    s.prune_type = "new_logits"  # new_logits (binary mask M) | gate_scores (G) in the EMA importance
    s.importance_threshold_factor = 0.6  # f0
    s.prune_interval = 30  # epochs
    s.min_experts = 1
    s.ema_decay = 0.9  # weight of the newest contribution (official convention)
    s.lr = 1e-3
    s.weight_decay = 0.0
    s.epochs = 1000
    s.batch_size = 0  # support instances per step (0 = full support); graph-level SGA population size

    s.edge = CN()
    s.edge.importance_threshold_factor = 0.3
    s.edge.prune_interval = 28

    s.graph = CN()
    s.graph.score_act = "softplus"
    s.graph.prune_type = "gate_scores"
    s.graph.imp_weight = 0.0
    s.graph.div_weight = 0.0
    s.graph.importance_threshold_factor = 0.4
    s.graph.prune_interval = 40
    s.graph.epochs = 200
    s.graph.batch_size = 32
