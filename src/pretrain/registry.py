from __future__ import annotations

from .task_base import PretrainTask

REGISTRY: dict[str, type[PretrainTask]] = {}
_METHODS_IMPORTED = False


def register(name: str):
    """Register a pretraining task class."""
    def decorator(cls: type[PretrainTask]):
        """Decorator to register the class."""
        REGISTRY[name.lower()] = cls
        cls.name = name.lower()
        return cls

    return decorator


def get_pretrain_task_class(name: str) -> type[PretrainTask] | None:
    """Return the registered class for ``name`` without instantiating it.

    Triggers lazy import of method modules so all ``@register`` decorators
    have fired.  Returns ``None`` if *name* is not found.
    """
    global _METHODS_IMPORTED
    if not _METHODS_IMPORTED:
        from . import methods  # noqa: F401 — triggers registration
        _METHODS_IMPORTED = True
    if not name:
        return None
    return REGISTRY.get(str(name).lower())


def variant_tag_for(cfg) -> str:
    """Resolve the method-variant tag for the given config.

    Returns ``""`` when the pretrain method is missing from cfg or not
    registered (a legitimate state during tooling imports and tests that
    never construct a runner). **Raises** if the registered method's
    ``variant_tag`` classmethod itself errors -- silently returning
    ``""`` would collapse distinct variant runs into the same checkpoint
    filename, which is the exact failure mode variant tags exist to
    prevent.
    """
    method = getattr(getattr(cfg, "pretrain", None), "method", "") or ""
    cls = get_pretrain_task_class(method)
    if cls is None:
        return ""
    # Intentionally *do not* call validate_cfg here. variant_tag_for is
    # invoked from non-pretraining code paths (finetune checkpoint
    # resolution, context/router generators) where the pretrain block is
    # used only for filename derivation. Validation is enforced at real
    # task-construction time via build_pretrain_task, so an invalid
    # pretrain config still fails fast when actually used for training.
    try:
        return cls.variant_tag(cfg) or ""
    except Exception as exc:
        raise RuntimeError(
            f"variant_tag({cls.__name__}) raised during run-name construction; "
            f"fix the classmethod or have it return '' explicitly. "
            f"Underlying error: {exc}"
        ) from exc


def build_pretrain_task(name: str, cfg) -> PretrainTask:
    """Build a pretraining task from the registry.

    Calls the class-level ``validate_cfg`` hook before instantiation so
    invalid method options fail fast with a clear ``ValueError`` instead
    of crashing mid-epoch.
    """
    cls = get_pretrain_task_class(name)
    if cls is None:
        raise ValueError(f"Unknown pretraining method: {name}")
    cls.validate_cfg(cfg)
    return cls(cfg)
