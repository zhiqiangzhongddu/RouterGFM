import unittest

from src.results.metric_policy import eval_metric


class MetricPolicyTest(unittest.TestCase):
    def test_policy_rules(self):
        self.assertEqual(eval_metric("qm7b", "graph", "regression"), "test_mae")
        self.assertEqual(eval_metric("cornell", "edge", "classification"), "test_auc")
        self.assertEqual(eval_metric("toxcast", "graph", "classification"), "test_auc")
        self.assertEqual(eval_metric("Tox21", "graph", "classification"), "test_auc")
        self.assertEqual(eval_metric("bace", "graph", "classification"), "test_acc")
        self.assertEqual(eval_metric("mnist", "graph", "classification"), "test_acc")
        self.assertEqual(eval_metric("photo", "node", "classification"), "test_acc")


if __name__ == "__main__":
    unittest.main()
