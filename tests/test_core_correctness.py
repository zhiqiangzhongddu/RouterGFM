from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.model import build_encoder_from_cfg
from src.model.nodeformer import NodeFormerConv, relu_kernel_transformation
from src.pretrain.augmentations import subgraph_sampling
from src.pretrain.methods.edge_pred import EdgePrediction, _unique_undirected_edges
from src.pretrain.trainer import PretrainRunner
from src.utils.monitoring import resolve_explicit_monitor_spec, supported_monitor_metric_values
from src.utils.naming import build_pretrain_run_name_from_cfg, model_variant_tag_for


def make_cfg() -> CN:
    return set_cfg(CN())


class CoreCorrectnessTest(unittest.TestCase):
    def test_test_split_monitors_are_rejected(self) -> None:
        for metric in ("test_loss", "test_acc", "test_auc", "test_mae"):
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, "held out"):
                    resolve_explicit_monitor_spec(
                        raw_monitor_metric=metric,
                        setting_name="train.monitor_metric",
                    )
                self.assertNotIn(metric, supported_monitor_metric_values())

    def test_edge_prediction_never_samples_reverse_of_directed_positive(self) -> None:
        task = EdgePrediction(make_cfg())
        data = Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[1], [0]], dtype=torch.long),
        )
        positives = _unique_undirected_edges(data.edge_index)
        for seed in range(20):
            torch.manual_seed(seed)
            negatives = task._sample_negatives_per_graph(
                data, positives, existing_edge_index=data.edge_index
            )
            unordered = {
                tuple(sorted((int(src), int(dst))))
                for src, dst in negatives.t().tolist()
            }
            self.assertNotIn((0, 1), unordered)

    def test_graphcl_subgraph_exact_target_and_incoming_frontier(self) -> None:
        data = Data(
            x=torch.arange(4, dtype=torch.float32).view(-1, 1),
            edge_index=torch.tensor([[0, 2], [1, 0]], dtype=torch.long),
        )
        with patch("torch.randint", return_value=torch.tensor([1])):
            sampled = subgraph_sampling(data, 0.75)
        self.assertEqual(sampled.num_nodes, 3)
        self.assertEqual(sampled.x.view(-1).tolist(), [0.0, 1.0, 2.0])

    def test_contrastive_loader_keeps_valid_partial_batch(self) -> None:
        runner = PretrainRunner.__new__(PretrainRunner)
        runner.task_cls = SimpleNamespace(min_graphs_per_batch=2, __name__="ContrastiveTask")
        runner.dataset = [
            Data(x=torch.randn(2, 3), edge_index=torch.tensor([[0, 1], [1, 0]]))
            for _ in range(3)
        ]
        runner.cfg = SimpleNamespace(pretrain=SimpleNamespace(batch_size=8, num_workers=0))
        runner.task_level_raw = "graph"
        runner.effective_task_level = "graph"
        loader, val_loader, test_loader = runner._make_full_dataset_loaders(induced=True)
        batches = list(loader)
        self.assertIsNone(val_loader)
        self.assertIsNone(test_loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].num_graphs, 3)

    def test_nodeformer_advertised_transforms(self) -> None:
        for rb_trans in ("sigmoid", "identity", "softplus", "relu"):
            with self.subTest(rb_trans=rb_trans):
                conv = NodeFormerConv(
                    in_channels=4,
                    out_channels=4,
                    num_heads=2,
                    kernel_transformation=relu_kernel_transformation,
                    projection_matrix_type=None,
                    use_gumbel=False,
                    rb_order=1,
                    rb_trans=rb_trans,
                    use_edge_loss=False,
                )
                conv.reset_parameters()
                z = torch.randn(1, 5, 4)
                edge_index = torch.tensor(
                    [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
                )
                out = conv(z, [(edge_index[0], edge_index[1])], tau=1.0)
                self.assertEqual(out.shape, z.shape)
                self.assertTrue(torch.isfinite(out).all())

    def test_project_nodeformer_rejects_unconstructed_higher_orders(self) -> None:
        cfg = make_cfg()
        cfg.model.name = "nodeformer"
        cfg.model.nodeformer.rb_order = 2
        with self.assertRaisesRegex(ValueError, "adjacency-power"):
            build_encoder_from_cfg(cfg, in_dim=cfg.model.in_dim)

    def test_nodeformer_rejects_invalid_rb_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "rb_order"):
            NodeFormerConv(4, 4, 2, rb_order=-1)
        with self.assertRaisesRegex(ValueError, "rb_trans"):
            NodeFormerConv(4, 4, 2, rb_trans="unknown")

    def test_sum_pooling_alternative_backbones(self) -> None:
        for model_name in ("h2gcn", "fagcn", "transformer", "nodeformer"):
            with self.subTest(model_name=model_name):
                cfg = make_cfg()
                cfg.model.name = model_name
                cfg.model.in_dim = 4
                cfg.model.hidden_dim = 8
                cfg.model.out_dim = 4
                cfg.model.num_layers = 2
                cfg.model.dropout = 0.0
                cfg.model.graph_pooling = "sum"
                cfg.model.nodeformer.use_gumbel = False
                cfg.model.nodeformer.use_edge_loss = False
                model = build_encoder_from_cfg(cfg, in_dim=4).eval()
                graphs = [
                    Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]])),
                    Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1], [1, 0]])),
                ]
                batch = Batch.from_data_list(graphs)
                with torch.no_grad():
                    node_repr, graph_repr = model(batch)
                expected = torch.stack(
                    [node_repr[batch.batch == graph_id].sum(dim=0) for graph_id in range(2)]
                )
                torch.testing.assert_close(graph_repr, expected)

    def test_per_backbone_batchnorm_flags_are_wired(self) -> None:
        for model_name in ("h2gcn", "fagcn"):
            cfg = make_cfg()
            cfg.model.name = model_name
            cfg.model.in_dim = 4
            getattr(cfg.model, model_name).use_batchnorm = True
            model = build_encoder_from_cfg(cfg, in_dim=4)
            self.assertTrue(any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules()))

    def test_nodeformer_and_warm_start_change_run_identity(self) -> None:
        cfg = make_cfg()
        cfg.model.name = "nodeformer"
        baseline = build_pretrain_run_name_from_cfg(cfg)
        cfg.pretrain.input_checkpoint = "outputs/pretrained_models/source/model.pt"
        warm = build_pretrain_run_name_from_cfg(cfg)
        self.assertNotEqual(warm, baseline)
        self.assertIn("wsmodel-", warm)
        cfg.model.nodeformer.rb_trans = "softplus"
        cfg.model.nodeformer.use_residual = False
        tag = model_variant_tag_for(cfg)
        self.assertIn("rbxsoftplus", tag)
        self.assertIn("nores", tag)


if __name__ == "__main__":
    unittest.main()
