"""META-DES (Cruz et al., 2015) on the matched frozen-expert pool (App. C; Tables 5, 6, 8).

Single-label classification only (node and graph classification).

* Pool C: the candidate experts (``baselines.candidate_rule``), each a frozen
  encoder with the infra head F_{a,e} fitted on S_a (no Bagging overproduction).
* DSEL = T_lambda = S_a with out-of-fold posteriors from the infra's
  cross-fitted heads, used leave-one-out (DESlib style).
* Region of competence: Euclidean K-NN in the label-free descriptor space
  z_a(x), standardized with support statistics; output profiles concatenate
  the pool's posteriors.
* Generalization (Algorithm 2 + DESlib): queries on which the whole pool
  agrees keep that label; otherwise experts with lambda(v) > threshold form
  C'(x) (all experts when none qualifies), combined by majority vote with the
  mean selected posterior as tie-break.

Queries are streamed per expert (two passes), so ``[|Q_a|, M, L]`` is never
materialised; the output-profile distances ``[|Q_a|, |S_a|]`` are.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from ...applications import derive_seed
from ...common import GRAPH_CLS, NODE_CLS
from ...descriptors import DescriptorStandardizer
from ..candidates import candidate_experts  # one candidate rule for every matched-pool runner
from .meta_features import build_meta_training_set, compute_meta_features, knn_indices
from .selector import fit_meta_selector

_LOG = "[RouterGFM][meta_des]"
_TIE_BREAK = 1e-3  # posterior weight in the vote score; < 1, so it only breaks ties
SUPPORTED_FAMILIES = (NODE_CLS, GRAPH_CLS)


def select_competent(comp: torch.Tensor, threshold: float) -> torch.Tensor:
    """``C'(x) = {e : comp > threshold}``; rows with no competent expert select the whole pool."""
    sel = comp > float(threshold)
    return sel | ~sel.any(dim=1, keepdim=True)


def vote_with_tiebreak(pred: torch.Tensor, post_sum: torch.Tensor, sel: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Scores ``[n, L]``: selected experts' votes + 1e-3 * their mean posterior.

    ``pred [n, M]`` are expert labels, ``post_sum [n, L]`` the sum of selected
    experts' posteriors, ``sel [n, M]`` the selection mask.
    """
    votes = torch.zeros(pred.size(0), int(num_classes), device=post_sum.device)
    votes.scatter_add_(1, pred.long(), sel.float())
    return votes + _TIE_BREAK * post_sum.float() / sel.sum(dim=1, keepdim=True).clamp_min(1)


class METADESRunner:
    """Matched-pool runner: ``fit()`` builds DSEL, trains lambda, predicts Q_a, and evaluates."""

    def __init__(self, cfg, app, infra):
        self.cfg, self.app, self.infra = cfg, app, infra
        self.mcfg = cfg.moe.routergfm.baselines.meta_des
        self.family = infra.task_family(app)
        if self.family not in SUPPORTED_FAMILIES:
            raise NotImplementedError(
                f"META-DES is evaluated on single-label classification only (App. C); "
                f"{app.key} is {self.family}."
            )
        self.device = torch.device(getattr(infra, "device", "cpu"))
        self.chunk = max(1, int(self.mcfg.query_chunk_size))
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch = None
        self.pool: List[str] = []
        self.selection = None
        self._pred = None
        self._stats: Dict[str, float] = {}

    # -- data ------------------------------------------------------------------
    def _expert_outputs(self, expert_id: str) -> Dict[str, torch.Tensor]:
        out = self.infra.expert_predictions(self.app, [expert_id])[expert_id]
        data = self.infra.data(self.app)
        if not (torch.equal(out["query_pos"], data.query_pos) and torch.equal(out["support_pos"], data.support_pos)):
            raise ValueError(f"{self.app.key}: predictions of {expert_id} are not aligned with S_a / Q_a.")
        return out

    def _chunks(self, n: int):
        return (slice(start, min(start + self.chunk, n)) for start in range(0, n, self.chunk))

    # -- phases ----------------------------------------------------------------
    def fit(self) -> None:
        """Pass 1 over experts (OOF posteriors, query labels/confidences, profile distances), lambda,
        then the query predictions and ``best_metrics``."""
        app, infra, m = self.app, self.infra, self.mcfg
        self.pool = candidate_experts(self.cfg, app, infra)
        data = infra.data(app)
        num_classes = int(data.num_classes)
        y_s = infra.support_labels(app).long().view(-1)
        n_s, n_q, M = y_s.numel(), int(data.query_pos.numel()), len(self.pool)
        z_s = infra.descriptors(app, "support").float()
        self._scaler = DescriptorStandardizer(clip=float(self.cfg.moe.routergfm.descriptors.clip)).fit(z_s)
        self._z_s = self._scaler.transform(z_s)

        oof = torch.empty(n_s, M, num_classes)
        self._q_pred = torch.empty(n_q, M, dtype=torch.int16)
        self._q_conf = torch.empty(n_q, M, dtype=torch.float16)
        # Output-profile distances up to a per-query constant: sum_e ||P_e(s)||^2 - 2 F_e(q) . P_e(s).
        self._op_dist = torch.zeros(n_q, n_s, device=self.device)
        for e, eid in enumerate(self.pool):
            out = self._expert_outputs(eid)
            oof[:, e] = out["support_oof_pred"]
            conf, label = out["pred"].max(dim=1)
            self._q_conf[:, e], self._q_pred[:, e] = conf.half(), label.to(torch.int16)
            p = out["support_oof_pred"].to(self.device)
            sq = p.pow(2).sum(dim=1)
            for c in self._chunks(n_q):
                self._op_dist[c].addmm_(out["pred"][c].to(self.device), p.T, alpha=-2.0).add_(sq)

        self._meta = build_meta_training_set(self._z_s, oof, y_s, int(m.k), int(m.kp), float(m.hc))
        self.selector, info = fit_meta_selector(
            self._meta.X, self._meta.y,
            hidden=int(m.meta_hidden), val_frac=float(m.meta_val_frac), patience=int(m.meta_patience),
            max_epochs=int(m.meta_max_epochs), seed=derive_seed(app.seed, "meta_des", app.data_key),
            device=self.device,
        )
        self._stats = {
            "meta_train_size": float(self._meta.X.size(0)),
            "meta_val_mse": float(info["val_mse"]),
            "consensus_kept_frac": self._meta.consensus_kept_frac,
        }
        print(
            f"{_LOG} {app.key}: pool {M}, |S_a| {n_s}, |Q_a| {n_q}, meta-set {self._meta.X.size(0)} rows "
            f"(consensus kept {self._meta.consensus_kept_frac:.2f}, fallback {self._meta.fallback}), "
            f"lambda val MSE {info['val_mse']:.4f} after {info['epochs']} epochs"
        )
        self._pred = None
        self.best_metrics = {f"test_{k}": float(v) for k, v in self.evaluate().items()}
        self.best_metrics.update(self._stats)

    @torch.no_grad()
    def predict_queries(self) -> torch.Tensor:
        """Class probabilities ``[|Q_a|, L]`` = normalized vote scores (their argmax is the META-DES label)."""
        if self._pred is not None:
            return self._pred
        app, infra, meta = self.app, self.infra, self._meta
        n_q, M = self._q_pred.shape
        num_classes = int(infra.data(app).num_classes)
        z_s = self._z_s.to(self.device)
        z_q = self._scaler.transform(infra.descriptors(app, "query").float())
        correct, post_true = meta.correct.to(self.device), meta.post_true.to(self.device)
        selector = self.selector.to(self.device)
        threshold = float(self.mcfg.selection_threshold)
        agree = (self._q_pred == self._q_pred[:, :1]).all(dim=1)  # DESlib all-agree shortcut
        sel = torch.empty(n_q, M, dtype=torch.bool)
        none_competent = torch.zeros(n_q, dtype=torch.bool)
        for c in self._chunks(n_q):
            theta = knn_indices(z_q[c].to(self.device), z_s, meta.k)
            phi = self._op_dist[c].topk(meta.kp, dim=1, largest=False).indices
            feats = compute_meta_features(correct, post_true, self._q_conf[c].to(self.device), theta, phi)
            comp = selector(feats).view(-1, M)
            sel[c] = select_competent(comp, threshold).cpu()
            none_competent[c] = (comp <= threshold).all(dim=1).cpu()
        sel[agree] = True  # the pool's unanimous label; posteriors averaged over the whole pool
        self.selection = sel  # C'(x) per query, [|Q_a|, M]
        self.selector = selector.cpu()
        self._op_dist = None  # the largest buffer; not needed after selection

        post_sum = torch.zeros(n_q, num_classes)
        for e, eid in enumerate(self.pool):  # pass 2: posterior tie-break term
            post_sum += sel[:, e : e + 1] * self._expert_outputs(eid)["pred"]
        scores = vote_with_tiebreak(self._q_pred, post_sum, sel, num_classes)
        self._pred = scores / scores.sum(dim=1, keepdim=True)

        dynamic = ~agree  # queries that went through dynamic selection (NaN rates when there are none)
        self._stats.update({
            "test_all_agree_rate": float(agree.float().mean()),
            "test_mean_ensemble_size": float(sel[dynamic].sum(dim=1).float().mean()),
            "test_fallback_rate": float(none_competent[dynamic].float().mean()),
        })
        return self._pred

    def evaluate(self) -> Dict[str, float]:
        """``infra.evaluate_outputs`` on Q_a (the only reader of query labels)."""
        return self.infra.evaluate_outputs(self.app, self.predict_queries())


__all__ = ["METADESRunner", "SUPPORTED_FAMILIES", "select_competent", "vote_with_tiebreak"]
