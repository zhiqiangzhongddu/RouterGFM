"""Shared CLI plumbing for workflow entrypoints (train / pretrain / finetune).

Each workflow's ``run.py`` exposes a thin ``build_<workflow>_cfg`` function
whose body is almost entirely boilerplate: call ``update_cfg``, record
explicit CLI keys, and validate that required dataset overrides were
provided. :func:`build_workflow_cfg` consolidates that pattern so there is
a single place to adjust CLI-validation policy.
"""

from __future__ import annotations

from typing import Callable, Iterable

from src.config import cfg as base_cfg, update_cfg
from src.utils.parsing import validate_required_dataset_overrides
from src.utils.save_results import extract_explicit_cfg_keys, set_explicit_cfg_keys


def build_workflow_cfg(
    *,
    workflow: str,
    argv: Iterable[str],
    dataset_name_key: str,
    task_level_key: str,
    run_tasks_tsv_key: str,
    empty_argv_hint: str,
    preprocess_argv: Callable[[list[str]], tuple[list[str], list[tuple[str, object]]]] | None = None,
):
    """Parse CLI overrides for a workflow and validate required inputs.

    Parameters
    ----------
    workflow : str
        Human-readable label used in error messages (e.g. ``"Train"``).
    argv : Iterable[str]
        Raw CLI tokens forwarded from ``sys.argv[1:]``.
    dataset_name_key : str
        Dotted cfg key that must be set (e.g. ``"train.dataset.name"``).
    task_level_key : str
        Dotted cfg key that must be set (e.g. ``"train.dataset.task_level"``).
    run_tasks_tsv_key : str
        Dotted cfg key guarding the TSV-batching short-circuit.
    empty_argv_hint : str
        Explanatory sentence appended to the "no CLI overrides" error.
    preprocess_argv : optional callable
        Hook for workflow-specific argv preprocessing.  Receives a list of
        tokens and returns ``(forwarded_argv, extra_explicit_entries)``
        where each entry is ``(dotted_key, value_to_assign)`` applied to
        the resulting cfg after ``update_cfg``.  Used by finetune to
        lift ``--fewshot`` into ``finetune.dataset.fixed_split``.
    """
    raw_argv = list(argv)
    if not raw_argv:
        raise ValueError(
            f"[{workflow}] No CLI overrides provided. {empty_argv_hint}"
        )

    if preprocess_argv is None:
        forwarded_argv = raw_argv
        extra_explicit: list[tuple[str, object]] = []
    else:
        forwarded_argv, extra_explicit = preprocess_argv(raw_argv)

    cfg = update_cfg(base_cfg, forwarded_argv)
    explicit_keys = extract_explicit_cfg_keys(forwarded_argv, flag_arity={"--config": 1})
    for key, value in extra_explicit:
        _assign_dotted(cfg, key, value)
        explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)

    if not _get_dotted(cfg, run_tasks_tsv_key):
        validate_required_dataset_overrides(
            cfg, forwarded_argv, workflow=workflow,
            dataset_name_key=dataset_name_key,
            task_level_key=task_level_key,
        )

    return cfg


def _assign_dotted(cfg, key: str, value) -> None:
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def _get_dotted(cfg, key: str):
    node = cfg
    for part in key.split("."):
        node = getattr(node, part)
    return node
