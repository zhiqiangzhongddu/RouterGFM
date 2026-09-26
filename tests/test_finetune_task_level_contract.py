from types import SimpleNamespace
import unittest

from src.finetune.methods.all_in_one import FinetuneAllInOne
from src.finetune.methods.gpf import FinetuneGPF
from src.finetune.methods.gppt import FinetuneGPPT
from src.finetune.methods.graphprompt import FinetuneGraphPrompt
from src.utils.dataset_helpers import (
    resolve_effective_task_level,
    resolve_loader_task_level,
    resolve_split_task_level,
)


def _cfg(task_level: str, induced: bool):
    return SimpleNamespace(
        finetune=SimpleNamespace(
            dataset=SimpleNamespace(task_level=task_level, induced=induced),
        )
    )


class TaskLevelContractTest(unittest.TestCase):
    def test_effective_task_level_resolves_induced_subgraphs(self):
        self.assertEqual(resolve_effective_task_level("node", True), "graph")
        self.assertEqual(resolve_effective_task_level("edge", True), "graph")
        self.assertEqual(resolve_effective_task_level("edge", False), "edge")
        self.assertEqual(resolve_effective_task_level("graph", False), "graph")

    def test_split_task_level_preserves_raw_level_for_induced_runs(self):
        self.assertEqual(resolve_split_task_level("edge", "graph", True), "edge")
        self.assertEqual(resolve_split_task_level("node", "graph", True), "node")
        self.assertEqual(resolve_split_task_level("node", "node", False), "node")

    def test_loader_task_level_preserves_induced_edge_loader_semantics(self):
        self.assertEqual(resolve_loader_task_level("edge", "graph", True), "edge")
        self.assertEqual(resolve_loader_task_level("node", "graph", True), "graph")

    def test_node_or_graph_prompt_methods_accept_induced_edge(self):
        for task_cls in (FinetuneGPF, FinetuneGPPT, FinetuneGraphPrompt):
            with self.subTest(method=task_cls.__name__):
                task_cls.validate_cfg(_cfg("edge", True))

    def test_node_or_graph_prompt_methods_reject_non_induced_edge(self):
        for task_cls in (FinetuneGPF, FinetuneGPPT, FinetuneGraphPrompt):
            with self.subTest(method=task_cls.__name__):
                with self.assertRaises(ValueError):
                    task_cls.validate_cfg(_cfg("edge", False))

    def test_node_or_graph_prompt_methods_keep_non_induced_node_support(self):
        for task_cls in (FinetuneGPF, FinetuneGPPT, FinetuneGraphPrompt):
            with self.subTest(method=task_cls.__name__):
                task_cls.validate_cfg(_cfg("node", False))

    def test_all_in_one_requires_graph_or_induced_subgraph_batches(self):
        FinetuneAllInOne.validate_cfg(_cfg("graph", False))
        FinetuneAllInOne.validate_cfg(_cfg("node", True))
        FinetuneAllInOne.validate_cfg(_cfg("edge", True))
        for task_level in ("node", "edge"):
            with self.subTest(task_level=task_level):
                with self.assertRaises(ValueError):
                    FinetuneAllInOne.validate_cfg(_cfg(task_level, False))


if __name__ == "__main__":
    unittest.main()
