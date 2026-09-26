"""Atomic induced-cache write tests (Phase 0 / Commit 2).

The pre-fix write was a raw ``torch.save`` that, when interrupted, left a
partial/corrupt ``.pt`` at the final path.  The fix writes to
``<path>.tmp.<pid>`` and renames on success, so either the final path
does not exist or it contains a complete payload.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.data_loader.induced_graphs import _save_induced_cache


class AtomicInducedCacheTest(unittest.TestCase):
    def test_successful_save_lands_at_final_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "foo.pt"
            payload = {"meta": {"k": 1}, "data": torch.zeros(3, 2)}
            _save_induced_cache(path, payload)
            self.assertTrue(path.is_file())
            # No tmp files left over.
            leftovers = [p for p in path.parent.iterdir() if ".tmp." in p.name]
            self.assertEqual(leftovers, [])
            # Payload round-trips.
            loaded = torch.load(path, weights_only=False)
            self.assertEqual(loaded["meta"], {"k": 1})
            self.assertTrue(torch.equal(loaded["data"], torch.zeros(3, 2)))

    def test_failed_save_does_not_leave_partial_at_final_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "foo.pt"
            payload = {"meta": {"k": 1}}

            real_save = torch.save

            def failing_save(obj, p, *args, **kwargs):
                # Write a truncated body, then raise.  Mirrors an OOM mid-save.
                real_save({"partial": True}, p)
                raise RuntimeError("simulated OOM mid-save")

            with mock.patch("src.data_loader.induced_graphs.torch.save",
                            side_effect=failing_save):
                with self.assertRaises(RuntimeError):
                    _save_induced_cache(path, payload)
            # Final path must not exist; tmp must be cleaned up.
            self.assertFalse(path.is_file())
            leftovers = list(path.parent.iterdir())
            self.assertEqual(leftovers, [])

    def test_concurrent_tmp_paths_do_not_collide(self):
        # Two writers at the same final path use different tmp names
        # because the tmp suffix includes PID.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "foo.pt"

            # Simulate a second process already holding a tmp file on
            # disk with its own PID.  Our write must not trip on it.
            other_pid_tmp = path.with_name(f"{path.name}.tmp.{os.getpid() + 1}")
            other_pid_tmp.parent.mkdir(parents=True, exist_ok=True)
            other_pid_tmp.write_bytes(b"not ours")

            payload = {"meta": {"k": 2}}
            _save_induced_cache(path, payload)

            # Our save succeeded, the other process's tmp is untouched.
            self.assertTrue(path.is_file())
            self.assertTrue(other_pid_tmp.is_file())


if __name__ == "__main__":
    unittest.main()
