from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_anygraph_cfg(cfg: CN) -> None:
    """Attach AnyGraph (MoE) pipeline config defaults to *cfg*.

    Must be called after :func:`set_moe_cfg` (so ``cfg.moe`` exists) and after
    ``cfg.data_preparation`` is set, because some defaults reference
    ``cfg.data_preparation.*_task_splits``.
    """
    # anygraph method options (grouped by stage)
    cfg.moe.anygraph = CN()
    # AnyGraph-specific expert pools (separate from the shared cfg.moe.expert_pools
    # used by other MoE methods). These TSVs carry a per-row task_level column
    # (dataset / task_level / task_type) that drives which level each dataset is
    # converted at. expert_pool keys in conversion/train/eval resolve here.
    cfg.moe.anygraph.expert_pools = CN()
    cfg.moe.anygraph.expert_pools.primary = "data/moe_anygraph_expert_pool_primary.tsv"  # 12 datasets
    # execution control
    cfg.moe.anygraph.execution = CN()
    cfg.moe.anygraph.execution.step = "all"  # conversion, train, eval, or all
    # shared paths/resources
    cfg.moe.anygraph.paths = CN()
    cfg.moe.anygraph.paths.dataset_root = "data/datasets"  # IcG dataset root
    cfg.moe.anygraph.paths.split_root = "data/splits"  # IcG split root
    cfg.moe.anygraph.paths.out_root = "data/anygraph_data"  # converted AnyGraph-format root
    # conversion stage
    cfg.moe.anygraph.conversion = CN()
    cfg.moe.anygraph.conversion.skip = False  # when True, reuse existing converted AnyGraph data
    cfg.moe.anygraph.conversion.task = "auto"  # auto, node, link, or all
    cfg.moe.anygraph.conversion.dataset = ""  # comma-separated dataset names, list, or empty
    cfg.moe.anygraph.conversion.expert_pool = ""  # comma-separated mix of expert-pool keys (primary), TSV paths, and dataset names; expanded + unioned with `dataset`
    cfg.moe.anygraph.conversion.test_dataset_file = ""  # optional extra held-out test-dataset TSV, unioned in (you can instead just list it in expert_pool/dataset)
    cfg.moe.anygraph.conversion.dataset_file = "data/available_node_datasets.tsv"  # fallback when dataset/expert_pool/test_dataset_file are all empty
    cfg.moe.anygraph.conversion.mask_col = 0  # split mask column index
    cfg.moe.anygraph.conversion.feat_reduction = False  # apply feature reduction during conversion
    cfg.moe.anygraph.conversion.l1_normalize_features = True  # row-wise L1 normalize link-task features
    cfg.moe.anygraph.conversion.feat_dim = 100  # feature SVD output dimension for link conversion
    cfg.moe.anygraph.conversion.seeds = []  # split seeds processed during AnyGraph conversion; empty means cfg.seeds
    cfg.moe.anygraph.conversion.edge_splits = [tuple(split) for split in cfg.data_preparation.edge_task_splits]  # edge splits to convert
    cfg.moe.anygraph.conversion.node_splits = [tuple(split) for split in cfg.data_preparation.node_task_splits]  # node splits to convert
    cfg.moe.anygraph.conversion.graph_splits = [tuple(split) for split in cfg.data_preparation.graph_task_splits]  # graph splits to convert
    cfg.moe.anygraph.conversion.edge_eval_payload_name = "agae_edge_eval_payload.pt"  # fixed val/test edge payload name
    cfg.moe.anygraph.conversion.node_output_feat_dim = 128  # output feature dim for node conversion
    cfg.moe.anygraph.conversion.graph_output_feat_dim = 128  # output feature dim for graph (super-node) conversion
    cfg.moe.anygraph.conversion.graph_filter_dir = "data/filters"  # empty-graph filter dir; must match split generation
    cfg.moe.anygraph.conversion.max_graphs = 0  # scale guard: cap graphs per graph dataset (0 = all); seeded + logged
    cfg.moe.anygraph.conversion.max_total_nodes = 0  # scale guard: cap ΣNᵢ per graph dataset (0 = unbounded); seeded + logged
    cfg.moe.anygraph.conversion.emit_node_val = True  # emit val_mat.pkl for node conversion
    # anygraph train stage
    cfg.moe.anygraph.train = CN()
    cfg.moe.anygraph.train.mode = "all"  # all (node+edge+graph, default) | node | link | both | graph
    cfg.moe.anygraph.train.epoch = 100  # AnyGraph epoch budget for the train step
    cfg.moe.anygraph.train.skip_if_exists = True  # skip training routes whose checkpoints already exist
    cfg.moe.anygraph.train.load_link_model = ""  # optional link checkpoint for resume training
    cfg.moe.anygraph.train.load_node_model = ""  # optional node checkpoint for resume training
    cfg.moe.anygraph.train.load_graph_model = ""  # optional graph checkpoint for resume training
    cfg.moe.anygraph.train.save_link_path = ""  # explicit link save-path tag; empty = auto from cfg (datasets/seeds/epoch)
    cfg.moe.anygraph.train.save_node_path = ""  # explicit node save-path tag; empty = auto from cfg (datasets/seeds/epoch)
    cfg.moe.anygraph.train.save_graph_path = ""  # explicit graph save-path tag; empty = auto from cfg (datasets/seeds/epoch)
    cfg.moe.anygraph.train.dataset = _default_dataset_cfg()
    cfg.moe.anygraph.train.dataset.name = None  # source dataset name(s); None means all converted datasets
    cfg.moe.anygraph.train.dataset.fixed_split = None  # fixed split used to resolve converted datasets across cfg.seeds
    cfg.moe.anygraph.train.link_dataset_setting = ""  # explicit AnyGraph link dataset_setting for training
    cfg.moe.anygraph.train.node_dataset_setting = ""  # explicit AnyGraph node dataset_setting for training
    cfg.moe.anygraph.train.graph_dataset_setting = ""  # explicit AnyGraph graph dataset_setting for training
    cfg.moe.anygraph.train.mixed_dataset_setting = ""  # shared dataset_setting split into node/link via conversion index
    # anygraph evaluation stage
    cfg.moe.anygraph.eval = CN()
    cfg.moe.anygraph.eval.mode = "all"  # all (node+edge+graph, default) | node | link | both | graph
    cfg.moe.anygraph.eval.load_link_model = ""  # explicit link checkpoint tag to evaluate
    cfg.moe.anygraph.eval.load_node_model = ""  # explicit node checkpoint tag to evaluate
    cfg.moe.anygraph.eval.load_graph_model = ""  # explicit graph checkpoint tag to evaluate
    cfg.moe.anygraph.eval.dataset = _default_dataset_cfg()
    cfg.moe.anygraph.eval.dataset.name = None  # source dataset name(s); None falls back to train.dataset.name or all converted datasets
    cfg.moe.anygraph.eval.dataset.fixed_split = None  # fixed split used to resolve target evaluation datasets across cfg.seeds
    cfg.moe.anygraph.eval.link_dataset_setting = ""  # explicit AnyGraph link dataset_setting for evaluation
    cfg.moe.anygraph.eval.node_dataset_setting = ""  # explicit AnyGraph node dataset_setting for evaluation
    cfg.moe.anygraph.eval.graph_dataset_setting = ""  # explicit AnyGraph graph dataset_setting for evaluation
    cfg.moe.anygraph.eval.mixed_dataset_setting = ""  # shared dataset_setting split into node/link via conversion index
    # prediction/evaluation stage
    cfg.moe.anygraph.prediction = CN()
    cfg.moe.anygraph.prediction.link = CN()
    cfg.moe.anygraph.prediction.link.eval_protocol = "agae"  # AnyGraph link eval protocol
    cfg.moe.anygraph.prediction.link.edge_eval_threshold_mode = "val_best_acc"  # val_best_acc or zero
    cfg.moe.anygraph.prediction.link.edge_eval_repeat_times = 5  # repeated edge eval rounds
    cfg.moe.anygraph.prediction.link.tst_epoch = 1  # passthrough to link runner when mode=link
    cfg.moe.anygraph.prediction.link.topk = 20  # passthrough to link runner when mode=link
    cfg.moe.anygraph.prediction.node = CN()
    cfg.moe.anygraph.prediction.node.tst_epoch = 1  # passthrough to node runner when mode=node
    cfg.moe.anygraph.prediction.node.assignment = "top1"  # passthrough to node runner when mode=node
    cfg.moe.anygraph.prediction.graph = CN()
    cfg.moe.anygraph.prediction.graph.tst_epoch = 1  # passthrough to graph runner when mode=graph
    cfg.moe.anygraph.prediction.graph.assignment = "top1"  # passthrough to graph runner when mode=graph
    # per-dataset prediction head (multilabel/regression families only; single-label
    # uses class-node link prediction and ignores these)
    cfg.moe.anygraph.prediction.graph.head_lr = 1e-3  # learning rate for the prediction head
    cfg.moe.anygraph.prediction.graph.head_weight_decay = 0.0  # weight decay for the prediction head
    cfg.moe.anygraph.prediction.graph.head_epoch = 200  # epochs to train the prediction head
    cfg.moe.anygraph.prediction.graph.head_hidden = 256  # hidden dim of the prediction head
    cfg.moe.anygraph.prediction.graph.head_layers = 2  # number of layers in the prediction head
    cfg.moe.anygraph.prediction.graph.head_dropout = 0.2  # dropout for the prediction head
    # output/report stage
    cfg.moe.anygraph.output = CN()
    cfg.moe.anygraph.output.link_csv = "outputs/anygraph/anygraph_link_eval.csv"
    cfg.moe.anygraph.output.node_csv = "outputs/anygraph/anygraph_node_eval.csv"
    cfg.moe.anygraph.output.graph_csv = "outputs/anygraph/anygraph_graph_eval.csv"
    cfg.moe.anygraph.output.report_csv = "outputs/anygraph/anygraph_report.csv"
    # optional passthrough args appended to link/node/graph runner scripts
    cfg.moe.anygraph.extra_args = []
