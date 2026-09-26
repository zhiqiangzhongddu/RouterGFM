from yacs.config import CfgNode as CN


def set_model_cfg(cfg: CN) -> None:
    """Attach model config defaults to *cfg*."""
    # General GNN options
    cfg.model = CN()
    cfg.model.name = "gcn"  # gcn, gin, gat, mlp, h2gcn, fagcn, transformer, nodeformer
    cfg.model.in_dim = 100  # inferred from dataset when 0 or None
    cfg.model.hidden_dim = 128  # hidden dimension
    cfg.model.out_dim = 128  # output dimension
    cfg.model.num_layers = 2  # number of GNN layers
    cfg.model.dropout = 0.5  # dropout rate
    cfg.model.activation = "relu"  # relu, gelu, prelu, elu, leaky_relu
    cfg.model.graph_pooling = "mean"  # mean, max, sum
    cfg.model.use_batchnorm = False  # apply BatchNorm1d after hidden layers (and layerwise InfoGraph cache)
    # FAGCN specific options
    cfg.model.fagcn = CN()
    cfg.model.fagcn.eps = 0.1  # initial epsilon value
    cfg.model.fagcn.use_batchnorm = False  # whether to use batchnorm
    # GAT specific options
    cfg.model.gat = CN()
    cfg.model.gat.heads = 8  # number of attention heads
    # H2GCN specific options
    cfg.model.h2gcn = CN()
    cfg.model.h2gcn.use_batchnorm = False  # whether to use batchnorm
    # NodeFormer specific options
    cfg.model.nodeformer = CN()
    cfg.model.nodeformer.heads = 4  # number of attention heads
    cfg.model.nodeformer.num_random_features = 30  # number of random features
    cfg.model.nodeformer.tau = 1.0  # softmax temperature
    cfg.model.nodeformer.use_layernorm = True  # whether to use layernorm
    cfg.model.nodeformer.use_gumbel = True  # whether to use gumbel softmax
    cfg.model.nodeformer.use_residual = True  # whether to use residual connections
    cfg.model.nodeformer.use_activation = True  # whether to use activation function
    cfg.model.nodeformer.use_jk = False  # whether to use jumpy knowledge connections
    cfg.model.nodeformer.num_gumbel_samples = 10  # number of gumbel samples
    cfg.model.nodeformer.rb_order = 1  # random basis order
    cfg.model.nodeformer.rb_trans = "sigmoid"  # transformation for random basis: sigmoid, softplus, relu
    cfg.model.nodeformer.use_edge_loss = False  # disable edge loss for encoder use cases
