from __future__ import annotations

from torch import nn

from src.finetune.task_base import _FinetuneBase

REGISTRY: dict[str, type[_FinetuneBase]] = {}
_METHODS_IMPORTED = False


def register(name: str):
    def decorator(cls: type[nn.Module]):
        REGISTRY[name.lower()] = cls
        cls.name = name.lower()
        return cls
    return decorator


def get_finetune_task_class(name: str) -> type[_FinetuneBase] | None:
    """Return the task class for *name* without instantiation.

    Triggers lazy import of method modules so all ``@register`` decorators
    have fired.  Returns ``None`` if *name* is not found.
    """
    global _METHODS_IMPORTED
    if not _METHODS_IMPORTED:
        from . import methods  # noqa: F401
        _METHODS_IMPORTED = True
    return REGISTRY.get(name.lower())


def build_finetune_task(name: str, cfg) -> _FinetuneBase:
    """Instantiate and return the task for *name*.

    Calls ``validate_cfg`` before instantiation as a second line of
    defense for direct factory callers (the runner already validates
    before skip-if-exists).
    """
    cls = get_finetune_task_class(name)
    if cls is None:
        raise ValueError(f"Unknown finetune method: {name}")
    cls.validate_cfg(cfg)
    return cls(cfg)
