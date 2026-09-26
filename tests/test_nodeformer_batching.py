"""Regression tests for NodeFormer batching and RNG hygiene.

The vendored kernelized attention summed keys over ALL nodes of the batched
tensor, so co-batched induced subgraphs attended to each other and every
prediction depended on batch composition. NodeFormerConv also called
``torch.manual_seed(data-dependent)`` inside ``forward``, hijacking the
process-wide RNG stream.
"""

import unittest

import torch

from src.model.nodeformer import (
    NodeFormer,
    NodeFormerConv,
    relu_kernel_transformation,
)


def _ring_edges(num_nodes: int, offset: int = 0) -> torch.Tensor:
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index + offset


class NodeFormerBatchingTest(unittest.TestCase):
    def _make_conv(self) -> NodeFormerConv:
        torch.manual_seed(0)
        conv = NodeFormerConv(
            in_channels=8,
            out_channels=8,
            num_heads=2,
            kernel_transformation=relu_kernel_transformation,
            projection_matrix_type=None,
            use_gumbel=False,
            rb_order=1,
            use_edge_loss=False,
        )
        conv.reset_parameters()
        conv.eval()
        return conv

    def test_batched_graphs_match_single_graph_outputs(self):
        conv = self._make_conv()
        n1, n2 = 7, 5
        torch.manual_seed(1)
        za = torch.randn(1, n1, 8)
        zb = torch.randn(1, n2, 8)
        ea = _ring_edges(n1)
        eb = _ring_edges(n2)

        with torch.no_grad():
            out_a = conv(za, [(ea[0], ea[1])], tau=1.0)
            out_b = conv(zb, [(eb[0], eb[1])], tau=1.0)

            z = torch.cat([za, zb], dim=1)
            edge_index = torch.cat([_ring_edges(n1), _ring_edges(n2, offset=n1)], dim=1)
            batch = torch.cat(
                [torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)]
            )
            out_batched = conv(z, [(edge_index[0], edge_index[1])], tau=1.0, batch=batch)
            out_unmasked = conv(z, [(edge_index[0], edge_index[1])], tau=1.0, batch=None)

        torch.testing.assert_close(out_batched[:, :n1], out_a, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out_batched[:, n1:], out_b, rtol=1e-4, atol=1e-5)
        # Sanity: without the batch vector the old cross-graph mixing happens.
        self.assertFalse(torch.allclose(out_unmasked[:, :n1], out_a, rtol=1e-4, atol=1e-5))

    def test_forward_does_not_disturb_global_rng(self):
        torch.manual_seed(0)
        model = NodeFormer(
            in_channels=6,
            hidden_channels=8,
            out_channels=4,
            num_layers=2,
            num_heads=2,
            rb_order=1,
            use_edge_loss=False,
        )
        model.eval()
        x = torch.randn(9, 6)
        edge_index = _ring_edges(9)

        torch.manual_seed(1234)
        expected = torch.rand(4)

        torch.manual_seed(1234)
        with torch.no_grad():
            model(x, [(edge_index[0], edge_index[1])], tau=1.0)
        observed = torch.rand(4)

        torch.testing.assert_close(observed, expected)

    def test_training_forward_backward_with_batch(self):
        torch.manual_seed(0)
        model = NodeFormer(
            in_channels=6,
            hidden_channels=8,
            out_channels=4,
            num_layers=2,
            num_heads=2,
            rb_order=1,
            use_gumbel=True,
            use_edge_loss=True,
        )
        model.train()
        n1, n2 = 6, 8
        x = torch.randn(n1 + n2, 6)
        edge_index = torch.cat([_ring_edges(n1), _ring_edges(n2, offset=n1)], dim=1)
        batch = torch.cat(
            [torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)]
        )
        out, link_losses = model(x, [(edge_index[0], edge_index[1])], tau=1.0, batch=batch)
        self.assertEqual(out.shape, (n1 + n2, 4))
        loss = out.sum() + sum(link_losses)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
