"""Deterministic train-only iteration for fitted target statistics."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator

import torch


def _loader_generators(loader) -> list[torch.Generator]:
    """Collect explicit loader/sampler generators without duplicates."""
    batch_sampler = getattr(loader, "batch_sampler", None)
    owners = (
        loader,
        getattr(loader, "sampler", None),
        batch_sampler,
        getattr(batch_sampler, "sampler", None),
    )
    generators = []
    seen = set()
    for owner in owners:
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator) and id(generator) not in seen:
            generators.append(generator)
            seen.add(id(generator))
    return generators


@contextmanager
def preserve_loader_rng(loader):
    """Keep a target-statistics pass invisible to later data ordering."""
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    generator_states = [
        (generator, generator.get_state().clone())
        for generator in _loader_generators(loader)
    ]
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is a project dependency
        np = None

    cuda_state = None
    if torch.cuda.is_initialized():
        cuda_state = torch.cuda.get_rng_state_all()

    try:
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        for generator, state in generator_states:
            generator.set_state(state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def iter_training_items_deterministically(train_loader) -> Iterator:
    """Yield the training dataset in stable index order when possible."""
    if hasattr(train_loader, "data") and not hasattr(train_loader, "dataset"):
        # ``SingleGraphDataLoader`` used by full-graph node tasks.
        yield train_loader.data
        return

    dataset = getattr(train_loader, "dataset", None)
    if (
        dataset is not None
        and hasattr(dataset, "__len__")
        and hasattr(dataset, "__getitem__")
    ):
        for index in range(len(dataset)):
            yield dataset[index]
        return

    # Lists and other small indexable test loaders do not expose ``dataset``.
    if hasattr(train_loader, "__len__") and hasattr(train_loader, "__getitem__"):
        for index in range(len(train_loader)):
            yield train_loader[index]
        return

    # Iterable-only fallback; ``preserve_loader_rng`` restores sampler state.
    yield from train_loader


__all__ = ["iter_training_items_deterministically", "preserve_loader_rng"]
