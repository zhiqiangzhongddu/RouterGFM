from yacs.config import CfgNode as CN


def _default_dataset_cfg() -> CN:
    """Build a dataset config block with common defaults."""
    ds = CN()
    ds.name = "cora"  # dataset name, None means to iterate over all available datasets
    ds.root = "data/datasets"  # root directory for datasets
    ds.available_node_datasets = "data/available_node_datasets.tsv"  # path to available node- and edge-level datasets list
    ds.available_graph_datasets = "data/available_graph_datasets.tsv"  # path to available graph-level datasets list
    ds.task_type = "none"  # classification or regression or none (none is used for pretraining runs on large datasets without available labels)
    ds.task_level = "none"  # node or graph or edge or none (none is used for pretraining runs on large datasets without available labels)
    ds.num_classes = None  # filled automatically if available
    ds.label_dim = None  # number of target dimensions when available (e.g., multi-task graph labels)
    ds.fixed_split = None  # when set, use fixed train/val/test split ratios (used by train/finetune/pretrain; not used by data_preparation)
    ds.num_splits = 5  # number of seeds per split definition; each split_def × seed pair produces one split
    ds.split_root = "data/splits"  # root directory for dataset splits
    ds.feat_reduction = True  # SVD feature reduction on node features
    ds.feat_reduction_svd_dim = 100  # target dimension for SVD feature reduction
    ds.feature_svd_dir = "data/feature_svd"  # output directory for feature SVD files
    ds.induced = True  # when True, operate on induced subgraphs
    ds.induced_min_size = 10  # min number of nodes for induced subgraphs
    ds.induced_max_size = 30  # max number of nodes for induced subgraphs
    ds.induced_max_hops = 5  # max hops to consider when building induced subgraphs
    ds.edge_max_size = 60  # max nodes in an induced edge-query subgraph
    ds.require_induced_cache_hit = False  # fail instead of building an induced cache at runtime
    ds.edge_level_max_num_nodes = 10000  # skip edge-level data preparation for node datasets above this node-count threshold; set <=0 to disable
    ds.induced_root = "data/induced_subgraphs"  # output directory for induced subgraphs
    ds.subgraph_svd = True  # generate subgraph SVD features during data prep
    ds.subgraph_svd_feat_dim = 100  # subgraph SVD feature dimension
    ds.subgraph_svd_struct_dim = 100  # subgraph SVD structure dimension
    ds.subgraph_svd_matrix = "adjacency"  # adjacency or laplacian
    ds.subgraph_svd_dir = "data/subgraph_svd"  # output directory for subgraph SVD files
    ds.graph_filter_dir = "data/filters"  # output directory for empty-graph filter masks

    return ds
