from yacs.config import CfgNode as CN


def set_save_results_cfg(cfg: CN) -> CN:
    """Result saving options."""
    cfg.save_results = CN()
    cfg.save_results.enabled = True  # save run summaries into outputs/<workflow>.tsv
    cfg.save_results.output_dir = "outputs/results"  # directory to save pretrain/train/finetune result tables
    cfg.save_results.save_skipped = False  # when True, also save rows for skipped runs
    cfg.save_results.explicit_keys = []  # internal: CLI keys explicitly provided for the current invocation

    return cfg
