from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_pretrain_cfg(cfg: CN) -> None:
    """Attach pretrain config defaults to *cfg*."""
    cfg.pretrain = CN()
    cfg.pretrain.method = "edge_pred"  # pretraining method: supervised, attr_masking, context_pred, dgi, edge_pred,  graphcl, infograph
    cfg.pretrain.epochs = 500  # maximum number of pretraining epochs
    cfg.pretrain.early_stopping = 50  # early stopping patience
    cfg.pretrain.lr = 1e-3  # learning rate
    cfg.pretrain.weight_decay = 0.0  # weight decay
    cfg.pretrain.batch_size = 128  # used for induced tasks and graph-level tasks
    cfg.pretrain.monitor_metric = "auto"  # pretrain monitor setting; auto is resolved by the shared monitoring policy
    cfg.pretrain.num_workers = 0  # number of data loading workers
    cfg.pretrain.checkpoint_dir = "outputs/pretrained_models"  # directory to save checkpoints
    cfg.pretrain.log_dir = "outputs/logs/pretrained_models"  # directory to save pretraining logs
    cfg.pretrain.run_tasks_tsv = False  # when True, pretrain on all datasets/tasks defined in tasks_tsv; set False to use pretrain.dataset settings
    cfg.pretrain.tasks_tsv = "slurm/pretrain.tsv"  # dataset/task definitions for slurm large scale pretraining; see src/pretrain/utils.py for supported row formats
    cfg.pretrain.skip_if_exists = True  # skip pretrain run if checkpoint already exists
    cfg.pretrain.scheduler = "none"  # learning rate scheduler: none, cosine, step
    cfg.pretrain.scheduler_step_size = 50  # step size for StepLR scheduler
    cfg.pretrain.scheduler_gamma = 0.5  # decay factor for StepLR scheduler
    cfg.pretrain.grad_clip = 0.0  # max gradient norm (0.0 = disabled)
    cfg.pretrain.input_checkpoint = ""  # optional path to a prior pretrain checkpoint; loads encoder weights before training (official edge_pred -> supervised chaining)
    # pretrain dataset options
    cfg.pretrain.dataset = _default_dataset_cfg()
    cfg.pretrain.dataset.fixed_split = (0.8, 0.1, 0.1)  # train, val, test split ratios
    # attr_masking: node attribute masking only.
    # Edge masking not implemented -- encoder family does not consume edge_attr.
    cfg.pretrain.attr_masking = CN()
    cfg.pretrain.attr_masking.mask_ratio = 0.15  # fraction of nodes whose features are replaced with the mask token
    cfg.pretrain.attr_masking.node_loss = "mse"  # "mse" (reconstruct full feature) or "ce" (classify x[:,0])
    cfg.pretrain.attr_masking.node_vocab_size = 0  # required (>= 2) when node_loss == "ce"
    # context_pred specific options
    cfg.pretrain.context_pred = CN()
    cfg.pretrain.context_pred.mode = "cbow"  # cbow or skipgram
    cfg.pretrain.context_pred.context_pooling = "mean"  # mean, sum, max
    cfg.pretrain.context_pred.neg_samples = 1  # number of negatives per positive pair
    cfg.pretrain.context_pred.context_size = 3  # l2 = (k-1) + context_size
    cfg.pretrain.context_pred.substruct_hops = 0  # 0 means use model.num_layers
    # dgi specific options
    cfg.pretrain.dgi = CN()
    cfg.pretrain.dgi.readout = "mean"  # readout pooling for graph summary: mean, add, max
    # edge_pred specific options
    cfg.pretrain.edge_pred = CN()
    cfg.pretrain.edge_pred.pos_edge_ratio = 1.0  # <1.0 to sample a subset of positive edges
    cfg.pretrain.edge_pred.pos_edge_max = 0  # >0 to cap the number of positive edges
    cfg.pretrain.edge_pred.neg_ratio = 1.0  # negatives per positive edge
    cfg.pretrain.edge_pred.forward_edge_ratio = 1.0  # <1.0 to sample edges used in message passing
    cfg.pretrain.edge_pred.forward_edge_max = 0  # >0 to cap edges used in message passing
    cfg.pretrain.edge_pred.use_mlp_scorer = False  # when True, use an MLP edge scorer instead of dot product
    # graphcl specific options
    cfg.pretrain.graphcl = CN()
    cfg.pretrain.graphcl.aug1 = "random"  # view 1 augmentation: dropN | permE | maskN | subgraph | random (see graphcl.py module docstring)
    cfg.pretrain.graphcl.aug2 = "random"  # view 2 augmentation: same options as aug1
    cfg.pretrain.graphcl.edge_remove_prob = 0.1  # edge removal probability
    cfg.pretrain.graphcl.permE_add_edges = False  # when True, permE also adds random edges (chem GraphCL variant)
    cfg.pretrain.graphcl.node_drop_prob = 0.1  # node dropping probability
    cfg.pretrain.graphcl.feature_mask_prob = 0.1  # feature masking probability
    cfg.pretrain.graphcl.use_subgraph_aug = True  # whether to use subgraph augmentation in addition to node/edge/feature perturbations
    cfg.pretrain.graphcl.subgraph_keep_ratio = 0.2  # node-KEEP ratio for GraphCL subgraph augmentation (note: sibling *_prob params are DROP rates)
    cfg.pretrain.graphcl.temperature = 0.1  # temperature for contrastive loss (GraphCL default)
    cfg.pretrain.graphcl.proj_hidden = 256  # projection head hidden dimension
    # infograph specific options
    # See src/pretrain/methods/infograph.py docstring for paper-vs-project divergences.
    cfg.pretrain.infograph = CN()
    cfg.pretrain.infograph.measure = "JSD"  # f-divergence measure (official InfoGraph uses JSD)
    cfg.pretrain.infograph.graph_pooling = "add"  # pooling used for InfoGraph global embedding
    cfg.pretrain.infograph.use_layerwise = True  # use concatenated per-layer node/graph reps (InfoGraph paper setting)
    cfg.pretrain.infograph.prior = False  # enable prior matching regularization
    cfg.pretrain.infograph.gamma = 0.1  # prior regularization coefficient
    # supervised specific options
    cfg.pretrain.supervised = CN()
