import copy
import unittest

from src.moe.anygraph import runtime
from src.moe.anygraph.naming import build_anygraph_route_run_name_from_cfg
from src.config import cfg as base_cfg


class AnyGraphRuntimePathsTest(unittest.TestCase):
    def test_package_root_resolves_to_anygraph_dir(self):
        self.assertTrue(
            runtime.ANYGRAPH_PACKAGE_ROOT.is_dir(),
            f"ANYGRAPH_PACKAGE_ROOT missing: {runtime.ANYGRAPH_PACKAGE_ROOT}",
        )
        self.assertEqual(runtime.ANYGRAPH_PACKAGE_ROOT.name, "anygraph")

    def test_vendored_entrypoints_exist(self):
        for path in (runtime.ANYGRAPH_LINK_MAIN, runtime.ANYGRAPH_NODE_MAIN):
            self.assertTrue(path.is_file(), f"Vendored entrypoint missing: {path}")

    def test_ensure_helper_raises_when_missing(self):
        missing = runtime.ANYGRAPH_PACKAGE_ROOT / "does_not_exist.py"
        with self.assertRaises(FileNotFoundError):
            runtime.ensure_anygraph_runtime_files_exist(missing)

    def test_runtime_env_sets_runtime_root(self):
        env = runtime.build_anygraph_runtime_env()
        self.assertEqual(env["ANYGRAPH_RUNTIME_ROOT"], str(runtime.ANYGRAPH_RUNTIME_ROOT))


class AnyGraphRunNameBuilderTest(unittest.TestCase):
    def _cfg(self, **train_overrides):
        cfg = copy.deepcopy(base_cfg)
        cfg.moe.anygraph.train.epoch = train_overrides.get("epoch", 100)
        return cfg

    def test_distinct_datasets_yield_distinct_tags(self):
        cfg = self._cfg()
        tag_a = build_anygraph_route_run_name_from_cfg(
            cfg, route="link", dataset_setting="cora_seed42_split-10-5-10",
            datasets=[], seeds=[42],
        )
        tag_b = build_anygraph_route_run_name_from_cfg(
            cfg, route="link", dataset_setting="arxiv_seed42_split-10-5-10",
            datasets=[], seeds=[42],
        )
        self.assertNotEqual(tag_a, tag_b)

    def test_epoch_change_forces_new_tag(self):
        args = dict(route="node", dataset_setting="cora_seed42_split-80-10-10",
                    datasets=[], seeds=[42])
        tag_e100 = build_anygraph_route_run_name_from_cfg(self._cfg(epoch=100), **args)
        tag_e200 = build_anygraph_route_run_name_from_cfg(self._cfg(epoch=200), **args)
        self.assertNotEqual(tag_e100, tag_e200)

    def test_seed_order_does_not_matter(self):
        args = dict(route="link", dataset_setting="cora_seed42_split-10-5-10", datasets=[])
        tag_a = build_anygraph_route_run_name_from_cfg(self._cfg(), seeds=[42, 0, 123], **args)
        tag_b = build_anygraph_route_run_name_from_cfg(self._cfg(), seeds=[123, 42, 0], **args)
        self.assertEqual(tag_a, tag_b)

    def test_long_dataset_setting_hashes_tail(self):
        long_setting = "+".join(f"d{i}_seed42_split-10-5-10" for i in range(30))
        tag = build_anygraph_route_run_name_from_cfg(
            self._cfg(), route="link", dataset_setting=long_setting,
            datasets=[], seeds=[42],
        )
        # The dataset token must stay bounded to keep filesystem tags sane.
        dataset_token = tag.split("_", 2)[2].rsplit("_seed", 1)[0]
        self.assertLessEqual(len(dataset_token), 64)
        self.assertIn("-h", dataset_token)

    def test_unknown_route_rejected(self):
        # "link", "node" and "graph" are all valid routes; only a genuinely
        # unknown route must raise.
        with self.assertRaises(ValueError):
            build_anygraph_route_run_name_from_cfg(
                self._cfg(), route="bogus", dataset_setting="x", datasets=[], seeds=[42],
            )

    def test_graph_route_accepted(self):
        tag = build_anygraph_route_run_name_from_cfg(
            self._cfg(), route="graph", dataset_setting="bbbp_seed42_split-80-10-10",
            datasets=[], seeds=[42],
        )
        self.assertTrue(tag.startswith("anygraph_graph_"))


if __name__ == "__main__":
    unittest.main()
