from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_train_cfg(cfg: CN) -> None:
    """Attach train config defaults to *cfg*."""
    cfg.train = CN()
    cfg.train.method = "supervised"  # training method (registry key)
    cfg.train.num_runs = 5  # number of training runs with different seeds
    cfg.train.epochs = 500  # maximum number of training epochs
    cfg.train.early_stopping = 50  # early stopping patience
    cfg.train.lr = 1e-3  # learning rate
    cfg.train.weight_decay = 0.0  # weight decay
    cfg.train.batch_size = 128  # used for induced tasks and graph-level tasks
    cfg.train.monitor_metric = "auto"  # canonical train monitor setting; auto is resolved by the shared monitoring policy
    cfg.train.num_workers = 0  # number of data loading workers
    cfg.train.checkpoint_dir = "outputs/trained_models"  # directory to save checkpoints
    cfg.train.log_dir = "outputs/logs/training_models"  # directory to save training logs
    cfg.train.skip_if_exists = True  # skip training run if checkpoint already exists
    cfg.train.scheduler = "none"  # learning rate scheduler: none, cosine, step
    cfg.train.scheduler_step_size = 50  # step size for StepLR scheduler
    cfg.train.scheduler_gamma = 0.5  # decay factor for StepLR scheduler
    cfg.train.grad_clip = 0.0  # max gradient norm (0.0 = disabled)
    cfg.train.run_tasks_tsv = False  # when True, train on all datasets/tasks defined in tasks_tsv; set False to use train.dataset settings
    cfg.train.tasks_tsv = "slurm/train.tsv"  # whitespace-delimited train rows: dataset model task_level task_type induced [fixed_split] [epochs] [batch]
    # train dataset options
    cfg.train.dataset = _default_dataset_cfg()
    cfg.train.dataset.fixed_split = (1, 0.0, 1.0)  # fixed train/val/test split ratios (overridden by specific pretrain/finetune settings)

    # Per-method hyperparameter blocks.  Kept as empty sub-nodes so future
    # methods land in the same ``cfg.train.<method>.*`` namespace used by
    # pretrain and finetune, instead of silently accruing top-level knobs.
    cfg.train.supervised = CN()
