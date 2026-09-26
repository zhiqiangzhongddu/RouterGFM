"""Frozen-encoder pretrained-weight load safety.

EdgePrompt and other frozen-encoder finetune methods depend on the
pretrained backbone weights being fully loaded before training starts.
A silent ``strict=False`` load that drops half the encoder keys would
leave the frozen backbone partially random and invalidate every
downstream metric without any visible error.

This module centralises the "did the frozen load actually land?" check
so every call site uses the same semantics.

Contract:
- ``pretrained_key_whitelist`` (optional attr on the encoder) = set of
  state_dict keys that MUST be supplied by the pretrained checkpoint.
  Defaults to "every encoder key" when the attr is missing / empty.
- ``prompt_only_keys`` (optional attr) = state_dict keys that are
  legitimately new (e.g. prompt tensors added by the prompt-aware
  encoder).  These are never counted against the load ratio.
- Keys that appear in ``unexpected_keys`` and are NOT in
  ``prompt_only_keys`` are reported as a hard error -- an unexpected
  key usually means the checkpoint and the encoder disagree about
  architecture.
"""

from __future__ import annotations

from typing import Iterable


def _as_frozenset(obj) -> frozenset[str]:
    try:
        return frozenset(str(k) for k in obj)
    except TypeError:
        return frozenset()


def check_frozen_encoder_load(
    *,
    encoder,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    min_match_ratio: float,
    require_frozen: bool,
) -> None:
    """Raise ``RuntimeError`` if a frozen-encoder load would leave too
    many whitelisted keys uninitialised, or if any non-prompt unexpected
    keys are present.

    When ``require_frozen`` is False, the check is a no-op; callers can
    still print the missing/unexpected keys as an advisory.
    """
    if not require_frozen:
        return

    missing = set(str(k) for k in missing_keys)
    unexpected = set(str(k) for k in unexpected_keys)

    whitelist = _as_frozenset(
        getattr(encoder, "pretrained_key_whitelist", None)
    )
    prompt_only = _as_frozenset(
        getattr(encoder, "prompt_only_keys", None)
    )

    if not whitelist:
        # Default: every encoder key must match.  Prompt-only keys are
        # still exempted from the "missing" count because they are
        # definitionally new.
        whitelist = frozenset(encoder.state_dict().keys()) - prompt_only

    missing_from_whitelist = missing & whitelist
    expected = len(whitelist)
    matched = expected - len(missing_from_whitelist)
    ratio = matched / max(expected, 1)

    if ratio < min_match_ratio:
        preview = sorted(missing_from_whitelist)[:5]
        raise RuntimeError(
            f"[Finetune] Frozen-encoder load failed safety check: "
            f"{matched}/{expected} whitelisted keys matched "
            f"(ratio={ratio:.3f} < threshold={min_match_ratio}). "
            f"Missing (first 5): {preview}. "
            f"A frozen encoder with unmatched keys is effectively random "
            f"and will invalidate downstream metrics."
        )

    unexpected_non_prompt = unexpected - prompt_only
    if unexpected_non_prompt:
        preview = sorted(unexpected_non_prompt)[:5]
        raise RuntimeError(
            f"[Finetune] Frozen-encoder load failed safety check: "
            f"unexpected non-prompt keys in checkpoint "
            f"(first 5): {preview}. "
            f"This usually means the pretrained architecture does not "
            f"match the finetune architecture."
        )
