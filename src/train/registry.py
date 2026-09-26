from __future__ import annotations

from src.train.task_base import TrainTask

REGISTRY: dict[str, type[TrainTask]] = {}
_METHODS_IMPORTED = False


def register(name: str):
    def decorator(cls: type[TrainTask]):
        REGISTRY[name.lower()] = cls
        cls.name = name.lower()
        return cls
    return decorator


def get_train_task_class(name: str) -> type[TrainTask] | None:
    """Return the task class for *name* without instantiation.

    Triggers lazy import of the task module so all ``@register`` decorators
    have fired.  Returns ``None`` if *name* is not found.
    """
    global _METHODS_IMPORTED
    if not _METHODS_IMPORTED:
        from . import methods  # noqa: F401 — triggers registration
        _METHODS_IMPORTED = True
    return REGISTRY.get(name.lower())


def build_train_task(name: str, cfg) -> TrainTask:
    """Instantiate and return the task for *name*.

    Calls ``validate_cfg`` before instantiation so invalid method
    options fail fast, matching the pretrain registry contract.
    """
    cls = get_train_task_class(name)
    if cls is None:
        raise ValueError(f"Unknown train method: {name}")
    cls.validate_cfg(cfg)
    return cls(cfg)
