"""Featureless graph datasets keep their published feature protocol.

qm7b has no native node features; every published pretrained/finetuned
artifact for it was built on the 1-dim EnsureFeatureTransform degree
feature. e8e09000 silently padded such datasets to feat_reduction_dim,
which made every existing qm7b encoder checkpoint unloadable (meta/loader
in_dim mismatch, runs die before training). Padding is now opt-in via
create_dataset(pad_featureless_features=True), used only by RouterGFM's
cross-dataset expert pipeline.
"""

import unittest
from pathlib import Path

from src.data_loader.datasets import create_dataset

COMMON = dict(
    root="data/datasets",
    task_level="graph",
    feat_reduction=True,
    feat_reduction_dim=100,
    feature_svd_dir="data/feature_svd",
    graph_filter_dir="data/filters",
)


def _requires_prepared(name):
    # Needs the real prepared dataset; skip instead of downloading it.
    present = (Path(COMMON["root"]) / name / "processed").is_dir()
    return unittest.skipUnless(present, f"prepared dataset '{name}' not found under {COMMON['root']}")


class FeaturelessFeatureProtocolTest(unittest.TestCase):
    @_requires_prepared("qm7b")
    def test_qm7b_default_keeps_published_1dim_protocol(self):
        ds = create_dataset(name="qm7b", **COMMON)
        self.assertEqual(ds[0].num_features, 1)

    @_requires_prepared("qm7b")
    def test_qm7b_pad_opt_in_serves_100dim(self):
        ds = create_dataset(name="qm7b", pad_featureless_features=True, **COMMON)
        self.assertEqual(ds[0].num_features, 100)

    @_requires_prepared("toxcast")
    def test_native_x_dataset_unaffected_by_default(self):
        ds = create_dataset(name="toxcast", **COMMON)
        self.assertEqual(ds[0].num_features, 100)


if __name__ == "__main__":
    unittest.main()
