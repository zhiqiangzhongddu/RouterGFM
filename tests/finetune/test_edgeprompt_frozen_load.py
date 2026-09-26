"""Frozen-encoder load safety tests (Phase 1 / Commit 3).

Covers:
- Partial load into a frozen encoder raises (default threshold 1.0).
- Lowered threshold permits partial load up to the configured ratio.
- Prompt-only keys that appear as ``unexpected_keys`` during load are
  whitelisted and do not trigger a failure.
- Unexpected non-prompt keys always fail (architecture mismatch signal).
- ``require_frozen=False`` is a no-op regardless of missing keys.
"""

from __future__ import annotations

import unittest

from torch import nn

from src.finetune.frozen_load import check_frozen_encoder_load


class _DummyEncoder(nn.Module):
    def __init__(self, whitelist=None, prompt_only=None):
        super().__init__()
        self.layer = nn.Linear(4, 4)
        if whitelist is not None:
            self.pretrained_key_whitelist = frozenset(whitelist)
        if prompt_only is not None:
            self.prompt_only_keys = frozenset(prompt_only)


class FrozenLoadSafetyTest(unittest.TestCase):
    def test_no_op_when_not_frozen(self):
        enc = _DummyEncoder()
        # All keys missing but require_frozen=False -> no error.
        check_frozen_encoder_load(
            encoder=enc,
            missing_keys=list(enc.state_dict().keys()),
            unexpected_keys=[],
            min_match_ratio=1.0,
            require_frozen=False,
        )

    def test_strict_default_rejects_any_missing_whitelisted_key(self):
        enc = _DummyEncoder()
        with self.assertRaises(RuntimeError) as ctx:
            check_frozen_encoder_load(
                encoder=enc,
                missing_keys=["layer.weight"],
                unexpected_keys=[],
                min_match_ratio=1.0,
                require_frozen=True,
            )
        self.assertIn("layer.weight", str(ctx.exception))

    def test_full_match_passes(self):
        enc = _DummyEncoder()
        check_frozen_encoder_load(
            encoder=enc,
            missing_keys=[],
            unexpected_keys=[],
            min_match_ratio=1.0,
            require_frozen=True,
        )

    def test_lowered_ratio_allows_partial_load(self):
        enc = _DummyEncoder(whitelist={"a", "b", "c", "d"})
        # 1/4 missing -> ratio = 0.75 -> passes at threshold 0.5.
        check_frozen_encoder_load(
            encoder=enc,
            missing_keys=["a"],
            unexpected_keys=[],
            min_match_ratio=0.5,
            require_frozen=True,
        )
        # 3/4 missing -> ratio = 0.25 -> fails at threshold 0.5.
        with self.assertRaises(RuntimeError):
            check_frozen_encoder_load(
                encoder=enc,
                missing_keys=["a", "b", "c"],
                unexpected_keys=[],
                min_match_ratio=0.5,
                require_frozen=True,
            )

    def test_prompt_only_keys_whitelisted_in_unexpected(self):
        enc = _DummyEncoder(
            whitelist={"layer.weight", "layer.bias"},
            prompt_only={"prompt.anchor_prompt.0"},
        )
        # Unexpected key is declared prompt-only -> accepted.
        check_frozen_encoder_load(
            encoder=enc,
            missing_keys=[],
            unexpected_keys=["prompt.anchor_prompt.0"],
            min_match_ratio=1.0,
            require_frozen=True,
        )

    def test_unexpected_non_prompt_keys_fail(self):
        enc = _DummyEncoder(
            whitelist={"layer.weight", "layer.bias"},
            prompt_only=set(),
        )
        with self.assertRaises(RuntimeError) as ctx:
            check_frozen_encoder_load(
                encoder=enc,
                missing_keys=[],
                unexpected_keys=["rogue.weight"],
                min_match_ratio=1.0,
                require_frozen=True,
            )
        self.assertIn("rogue.weight", str(ctx.exception))

    def test_missing_prompt_only_key_does_not_count_against_ratio(self):
        # Whitelist auto-derivation: when no explicit whitelist, prompt_only
        # keys are subtracted from "all encoder keys" so they don't trigger
        # a failure when missing.
        enc = _DummyEncoder(prompt_only={"layer.bias"})
        # "layer.bias" is prompt-only -> whitelist = {"layer.weight"} only.
        # Missing layer.bias is fine.
        check_frozen_encoder_load(
            encoder=enc,
            missing_keys=["layer.bias"],
            unexpected_keys=[],
            min_match_ratio=1.0,
            require_frozen=True,
        )
        # But missing the non-prompt-only key still fails.
        with self.assertRaises(RuntimeError):
            check_frozen_encoder_load(
                encoder=enc,
                missing_keys=["layer.weight"],
                unexpected_keys=[],
                min_match_ratio=1.0,
                require_frozen=True,
            )


if __name__ == "__main__":
    unittest.main()
