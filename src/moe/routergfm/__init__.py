"""RouterGFM: inductive routing and local expertise transfer over a frozen expert pool.

The pipeline is imported only when :func:`run_routergfm` is called, so importing
a submodule (e.g. ``src.moe.routergfm.losses``) stays light.
"""


def run_routergfm(cfg, **kwargs) -> int:
    """Run the stage named by ``cfg.moe.routergfm.task`` (see :mod:`.run`)."""
    from .run import run_routergfm as _run

    return _run(cfg, **kwargs)


__all__ = ["run_routergfm"]
