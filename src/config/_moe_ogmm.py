from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_ogmm_cfg(cfg: CN) -> None:
    """Attach OGMM (Wang et al., ICLR 2026): out-of-distribution graph models merging defaults.

    OGMM builds its own inventory: the target support is split into edge-density
    domains, and one dense 2-layer GCN / GAT / GIN expert is trained per domain.
    Each expert then inverts label-conditional graphs (stage 1). A noisy top-k
    gate plus classifier masks merge the experts on those generated graphs only,
    i.e. source-free (stage 2). Values marked PROPOSED are not given by the
    paper (see the OGMM spec).

    Owns the ``cfg.moe.ogmm`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.ogmm = CN()

    # ------------------------------------------------------------------ #
    # Target dataset / task
    # ------------------------------------------------------------------ #
    cfg.moe.ogmm.dataset = _default_dataset_cfg()
    cfg.moe.ogmm.dataset.task_level = "node"  # node, edge, or graph (node/edge must be induced)
    cfg.moe.ogmm.dataset.task_type = "none"  # classification | regression | none (auto-detect)
    cfg.moe.ogmm.dataset.induced = True  # node/edge tasks are consumed as induced subgraphs
    cfg.moe.ogmm.dataset.fixed_split = (5, 0.0, 1.0)  # few-shot support; queries = test split
    cfg.moe.ogmm.in_dim = 0  # input feature dim; inferred from dataset when 0/None

    # ------------------------------------------------------------------ #
    # Stage 0: own inventory (density domains x architectures)
    # ------------------------------------------------------------------ #
    cfg.moe.ogmm.num_domains = 2  # paper A/B source domains (support split by edge density)
    cfg.moe.ogmm.expert_archs = ["gcn", "gat", "gin"]  # paper backbones
    cfg.moe.ogmm.expert_hidden_dim = 32  # paper: 2-layer, 32-dim experts
    cfg.moe.ogmm.expert_dropout = 0.5  # PROPOSED (paper silent)
    cfg.moe.ogmm.expert_epochs = 500  # PROPOSED (repo train workflow default); train-loss monitor
    cfg.moe.ogmm.expert_lr = 1e-3  # PROPOSED (repo train workflow default, Adam)

    # ------------------------------------------------------------------ #
    # Stage 1: label-conditional graph generation (Eqs. 7-11)
    # ------------------------------------------------------------------ #
    cfg.moe.ogmm.gen_epochs = 200  # paper App. B.2 (AdamW)
    cfg.moe.ogmm.gen_lr = 1e-2  # PROPOSED
    cfg.moe.ogmm.gen_num_graphs = 64  # PROPOSED, generated graphs per expert
    cfg.moe.ogmm.gen_edge_hidden_dim = 64  # PROPOSED, width of the 3-layer edge MLP
    cfg.moe.ogmm.gen_gumbel_tau = 0.5  # PROPOSED, relaxed-Bernoulli temperature (Eq. 8)
    cfg.moe.ogmm.gen_edge_threshold = 0.5  # PROPOSED, hard edges of the stored graphs

    # ------------------------------------------------------------------ #
    # Stage 2: fine-tuned MoE merging (Eqs. 12-20)
    # ------------------------------------------------------------------ #
    cfg.moe.ogmm.merge_epochs = 20  # paper App. B.2 (AdamW); last epoch kept
    cfg.moe.ogmm.merge_lr = 1e-3  # PROPOSED
    cfg.moe.ogmm.top_k = 2  # paper grid {1..5}; 2-4 best (Fig. 5)
    cfg.moe.ogmm.lambda_gate = 0.1  # paper Fig. 11 best cell (MUTAG, k=2)
    cfg.moe.ogmm.lambda_mask = 0.01  # paper Fig. 11 best cell (MUTAG, k=2)
    cfg.moe.ogmm.gamma_p = 0.5  # PROPOSED (shifts the loss value only under the literal Eq. 19)
    cfg.moe.ogmm.gamma_v = 0.1  # PROPOSED (shifts the loss value only under the literal Eq. 19)

    # ------------------------------------------------------------------ #
    # Run options
    # ------------------------------------------------------------------ #
    cfg.moe.ogmm.batch_size = 32  # expert training, merging and query batches
    cfg.moe.ogmm.num_workers = 0  # data loading workers
    cfg.moe.ogmm.num_runs = 5  # number of runs with different seeds
    cfg.moe.ogmm.checkpoint_dir = "outputs/moe/ogmm/checkpoints"  # directory to save checkpoints
    cfg.moe.ogmm.log_dir = "outputs/moe/ogmm/logs"  # directory to save run logs
    cfg.moe.ogmm.prediction_dir = "outputs/moe/ogmm/predictions"  # per-query predictions (Brier inputs)
    cfg.moe.ogmm.skip_if_exists = True  # skip run if checkpoint already exists

    # ------------------------------------------------------------------ #
    # Batch execution (TSV)
    # ------------------------------------------------------------------ #
    cfg.moe.ogmm.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.ogmm.tasks_tsv = "slurm/moe.ogmm.all.tsv"  # header-based TSV of OGMM tasks
