from yacs.config import CfgNode as CN


def set_moe_cfg(cfg: CN) -> None:
    """Attach shared mixture-of-experts (MoE) config defaults to *cfg*.

    Owns the base ``cfg.moe`` node and keys shared across MoE methods. Each
    method's own subtree lives in a sibling module and is attached afterwards:
    ``cfg.moe.anygraph`` via :func:`._moe_anygraph.set_anygraph_cfg` and
    ``cfg.moe.routergfm`` via :func:`._moe_routergfm.set_routergfm_cfg`.
    """
    cfg.moe = CN()
    cfg.moe.method = ""  # MoE method name (dispatch key); set explicitly in run_moe.py
    # Expert pools for the non-AnyGraph MoE methods. Each value is a path to a
    # comment-tolerant TSV whose first column lists the pool's dataset names.
    # AnyGraph uses its own pools (cfg.moe.anygraph.expert_pools, the
    # data/moe_anygraph_*.tsv files) because it also needs the per-row
    # task_level column. Override per-run with e.g.
    # `moe.expert_pools.primary data/other.tsv`.
    cfg.moe.expert_pools = CN()
    cfg.moe.expert_pools.primary = "data/moe_expert_pool_primary.tsv"  # 12 datasets
