from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg

# Official real-world (GOOD) training shift list, ``scripts/train_moe_good.sh``:
# 5 single transforms + 9 pairs; ``-`` composes left to right.
_GOOD_PAIRED = "/".join([
    "noisy_node_feat", "add_edge", "drop_edge", "drop_node", "random_subgraph",
    "noisy_node_feat-add_edge", "noisy_node_feat-drop_edge", "noisy_node_feat-random_subgraph",
    "noisy_node_feat-drop_node", "add_edge-drop_node", "add_edge-random_subgraph",
    "drop_edge-drop_node", "drop_edge-random_subgraph", "drop_node-random_subgraph",
])


def set_graphmetro_cfg(cfg: CN) -> None:
    """Attach GraphMETRO (Wu et al., NeurIPS 2024): mixture of aligned experts under shift defaults.

    A gating GNN predicts which stochastic shift components (transforms) are
    present in an instance; ``K+1`` independent GNN experts (expert 0 is the
    reference) are mixed by the softmax gate and aligned to the reference
    expert's embedding of the clean instance. Trained end-to-end on the target
    support; node / edge tasks are consumed as induced subgraph instances.

    Owns the ``cfg.moe.graphmetro`` subtree. Attached after the shared ``cfg.moe`` node.
    """
    cfg.moe.graphmetro = CN()

    # ------------------------------------------------------------------ #
    # Target dataset / task
    # ------------------------------------------------------------------ #
    cfg.moe.graphmetro.dataset = _default_dataset_cfg()
    cfg.moe.graphmetro.dataset.task_level = "node"  # node, edge, or graph (node/edge must be induced)
    cfg.moe.graphmetro.dataset.task_type = "none"  # classification | regression | none (auto-detect)
    cfg.moe.graphmetro.dataset.induced = True  # node/edge tasks are consumed as induced subgraphs
    cfg.moe.graphmetro.dataset.fixed_split = (5, 0.0, 1.0)  # few-shot support; queries = test split
    cfg.moe.graphmetro.in_dim = 0  # input feature dim; inferred from dataset when 0/None

    # ------------------------------------------------------------------ #
    # Architecture (paper Table 3, real-world: GCN node / vGIN graph, 3 x 300, dropout 0.5)
    # ------------------------------------------------------------------ #
    cfg.moe.graphmetro.backbone = "gcn"  # gate + expert GNN type; TSV sets gin for graph datasets
    cfg.moe.graphmetro.num_layers = 3
    cfg.moe.graphmetro.hidden_dim = 300
    cfg.moe.graphmetro.dropout = 0.5
    cfg.moe.graphmetro.use_batchnorm = True  # GOOD encoders use batch norm
    cfg.moe.graphmetro.graph_pooling = "mean"  # graph instances only; node/edge use target/endpoint readout

    # ------------------------------------------------------------------ #
    # Shift components and objective (Eq. 3)
    # ------------------------------------------------------------------ #
    cfg.moe.graphmetro.shift_train_types = _GOOD_PAIRED  # "/"-separated training shifts; "-" composes
    cfg.moe.graphmetro.shift_p = 0.5  # probability / ratio of the edge, node and feature transforms
    cfg.moe.graphmetro.shift_k = 2  # hops of random_subgraph
    cfg.moe.graphmetro.num_shift_samples = 3  # shifts sampled (with replacement) per mini-batch
    cfg.moe.graphmetro.gate_pos_weight = 4.0  # official BCE pos_weight of the gate loss (paper silent)
    cfg.moe.graphmetro.align_lambda = 1.0  # paper value (the released script effectively uses 0)

    # ------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------ #
    cfg.moe.graphmetro.moe_lr = 1e-2  # gate + experts (paper: 1e-2 node, 1e-3 graph)
    cfg.moe.graphmetro.classifier_lr = 1e-4  # shared classifier head
    cfg.moe.graphmetro.weight_decay = 0.0
    cfg.moe.graphmetro.epochs = 100
    cfg.moe.graphmetro.early_stopping = 0  # early stopping patience (0 disables)
    cfg.moe.graphmetro.batch_size = 32
    cfg.moe.graphmetro.num_workers = 0
    cfg.moe.graphmetro.monitor_metric = "auto"  # auto | disabled | explicit metric (few-shot -> train_loss)
    cfg.moe.graphmetro.grad_clip = 0.0  # max gradient norm (0.0 = disabled)
    cfg.moe.graphmetro.num_runs = 5  # number of runs with different seeds
    cfg.moe.graphmetro.checkpoint_dir = "outputs/moe/graphmetro/checkpoints"
    cfg.moe.graphmetro.log_dir = "outputs/moe/graphmetro/logs"
    cfg.moe.graphmetro.prediction_dir = "outputs/moe/graphmetro/predictions"  # per-query predictions (Brier inputs)
    cfg.moe.graphmetro.skip_if_exists = True  # skip run if checkpoint already exists

    # ------------------------------------------------------------------ #
    # Batch execution (TSV)
    # ------------------------------------------------------------------ #
    cfg.moe.graphmetro.run_tasks_tsv = False  # when True, run all rows in tasks_tsv instead of dataset settings
    cfg.moe.graphmetro.tasks_tsv = "slurm/moe.graphmetro.all.tsv"  # header-based TSV of GraphMETRO tasks
