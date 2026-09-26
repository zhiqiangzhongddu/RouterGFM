from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_data_preparation_cfg(cfg: CN) -> CN:
    """Dataset preparation options."""
    cfg.data_preparation = CN()
    cfg.data_preparation.dataset = _default_dataset_cfg()
    cfg.data_preparation.target_datasets = None  # required: "data_name" -> one dataset; "data_name1,data_name2" -> multiple datasets; existing/path-like file -> read names from file
    cfg.data_preparation.task_level_override = ""  # optional node/edge/graph override for explicitly selected targets
    cfg.data_preparation.node_task_splits = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.8), (100, 0.0, 1.0), (5, 0.0, 1.0)]
    cfg.data_preparation.graph_task_splits = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.8), (100, 0.0, 1.0), (5, 0.0, 1.0)]
    cfg.data_preparation.edge_task_splits = [(0.05, 0.1, 0.1), (0.1, 0.05, 0.1)]
    cfg.data_preparation.generate_edge_level = True  # whether to generate edge-level information for node-level datasets
    cfg.data_preparation.batch_size = 64  # batch size for split-stat materialization during data prep
    cfg.data_preparation.num_workers = 0  # data loading workers for split-stat materialization during data prep
    cfg.data_preparation.summary_file = "data/summary.tsv"  # output file for dataset summary information

    return cfg
