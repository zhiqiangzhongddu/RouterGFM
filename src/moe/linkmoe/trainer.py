"""Link-MoE runner: per-seed two-step pipeline (Ma et al., NeurIPS 2024, Alg. 1).

1. Train every expert independently on the train pairs (val-AUC selection)
   and score val and test pairs.
2. Compute gate inputs for val/test pairs on the eval context graph (pair
   features ``x_i * x_j`` and raw structural heuristics), split the val
   pairs per class into gate-train / gate-val, train the gate on gate-train
   (best gate-val AUC epoch), then evaluate the mixture on the test pairs.

Test pairs are used only for the final evaluation (and logged diagnostics).
"""

from __future__ import annotations

import os

import torch

from src.moe.identity import behavior_fingerprint
from src.utils.checkpoint import save_checkpoint, save_training_log
from src.utils.dataset_helpers import is_few_shot_split, shared_induced_root, shared_split_root
from src.utils.naming import format_split_for_name
from src.utils.parsing import resolve_task_type, resolve_workflow_split
from src.utils.random import set_seed

from .data import build_link_views, build_seal_view, stratified_split
from .expert_training import probability_metrics, train_full_graph_expert, train_seal_expert
from .experts import EXPERT_BUILDERS, SUBGRAPH_EXPERTS
from .gate import LinkMoEGate, gate_loss, mixture_probability
from .heuristics import HEURISTIC_NAMES, pair_heuristics, symmetric_csr

_DEFAULT_SPLIT = (0.1, 0.05, 0.1)
_CN_BUCKETS = (("0", 0.0, 1.0), ("1", 1.0, 2.0), ("2", 2.0, 3.0), (">=3", 3.0, float("inf")))


def validate_link_task(ds_cfg) -> tuple:
    """Return the edge split of a link-prediction task; raise for any other task."""
    level = str(getattr(ds_cfg, "task_level", "")).lower()
    if level != "edge":
        raise ValueError(
            f"[LinkMoE] Link-MoE only supports link prediction (task_level='edge'); got task_level='{level}'."
        )
    task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
    if task_type != "classification":
        raise ValueError(f"[LinkMoE] Link prediction is binary classification; got task_type='{task_type}'.")
    split = resolve_workflow_split(getattr(ds_cfg, "fixed_split", None), default=_DEFAULT_SPLIT)
    if is_few_shot_split(split):
        raise ValueError(f"[LinkMoE] Edge tasks need a positive-edge ratio split, got {tuple(split)}.")
    return tuple(float(v) for v in split)


class LinkMoERunner:
    """Train and evaluate Link-MoE for one seed using ``cfg.moe.linkmoe``."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lcfg = cfg.moe.linkmoe
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)

        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.train_history: list[dict] = []
        self.dataset_meta: dict = {}

        ds_cfg = self.lcfg.dataset
        self.split = validate_link_task(ds_cfg)
        self.expert_names = self._resolve_experts()
        self.run_name = self._build_run_name()
        self.run_group = f"{ds_cfg.name}-edge"
        self.run_dir = os.path.join(self.lcfg.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        if bool(self.lcfg.skip_if_exists) and os.path.isfile(self._checkpoint_path()):
            self._skip_due_to_existing_checkpoint = True
            print(f"[LinkMoE] Checkpoint already exists, skipping: {self._checkpoint_path()}")

    # ------------------------------------------------------------------ #
    # Identity / paths
    # ------------------------------------------------------------------ #
    def _resolve_experts(self) -> list[str]:
        names = [str(n).strip().lower() for n in self.lcfg.experts if str(n).strip()]
        unknown = sorted(set(names) - set(EXPERT_BUILDERS))
        if unknown or not names or len(set(names)) != len(names):
            raise ValueError(
                f"[LinkMoE] experts must be distinct names from {sorted(EXPERT_BUILDERS)}; got {list(self.lcfg.experts)}."
            )
        return names

    def _build_run_name(self) -> str:
        gate = self.lcfg.gate
        fingerprint = behavior_fingerprint(
            self.lcfg,
            external_behavior={
                "shared_split_root": shared_split_root(self.cfg),
                "shared_induced_root": shared_induced_root(self.cfg, self.lcfg.dataset.induced_root),
            },
        )
        parts = [
            "linkmoe",
            self.lcfg.dataset.name,
            format_split_for_name(self.split),
            "taskedge",
            "x" + "-".join(self.expert_names),
            f"gh{gate.hidden_dim}",
            f"gl{gate.num_layers}",
            f"gp{gate.num_layers_predictor}",
            f"ge{gate.epochs}",
            f"glr{float(gate.lr):g}",
            f"nw{float(gate.neg_loss_weight):g}",
            f"cfg{fingerprint}",
            f"seed{self.cfg.seed}",
        ]
        return "_".join(str(p) for p in parts if p not in ("", None))

    def _checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}.pt")

    def get_checkpoint_path_for_metrics(self) -> str:
        return self._checkpoint_path()

    def _log_path(self) -> str:
        log_dir = getattr(self.lcfg, "log_dir", "")
        if log_dir:
            return os.path.join(log_dir, self.run_group, f"{self.run_name}_log.json")
        return os.path.join(self.run_dir, f"{self.run_name}_log.json")

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def _train_experts(self, views, seed: int) -> list:
        in_dim = int(views.x.size(1))
        seal_view = None
        scores = []
        for name in self.expert_names:
            set_seed(seed)
            expert = EXPERT_BUILDERS[name](self.lcfg, in_dim)
            if name in SUBGRAPH_EXPERTS:
                seal_view = seal_view or build_seal_view(self.cfg, seed, views)
                scores.append(train_seal_expert(expert, seal_view, views, self.cfg, self.device, seed))
            else:
                scores.append(train_full_graph_expert(name, expert, views, self.cfg, self.device, seed))
        return scores

    def _gate_inputs(self, views) -> tuple[dict, dict]:
        """Pair features and heuristics of val/test pairs on the eval context graph ``C``."""
        A = symmetric_csr(views.context_edge_index, views.num_nodes)
        eval_pairs = torch.cat([views.pairs["val"], views.pairs["test"]], dim=1)
        struct = pair_heuristics(
            A,
            eval_pairs,
            katz_beta=float(self.lcfg.katz_beta),
            ppr_damping=float(self.lcfg.ppr_damping),
            ppr_tol=float(self.lcfg.ppr_tol),
            ppr_max_iter=int(self.lcfg.ppr_max_iter),
            device=self.device,
        )
        n_val = views.pairs["val"].size(1)
        feats = {s: views.x[views.pairs[s][0]] * views.x[views.pairs[s][1]] for s in ("val", "test")}
        return feats, {"val": struct[:n_val], "test": struct[n_val:]}

    def _train_gate(self, feats, structs, probs, labels, gate_train, gate_val, seed: int):
        gcfg = self.lcfg.gate
        dev = self.device
        set_seed(seed)
        gate = LinkMoEGate(
            feat_dim=int(feats["val"].size(1)),
            struct_dim=len(HEURISTIC_NAMES),
            hidden_dim=int(gcfg.hidden_dim),
            num_layers=int(gcfg.num_layers),
            num_layers_predictor=int(gcfg.num_layers_predictor),
            num_experts=len(self.expert_names),
            dropout=float(gcfg.dropout),
        ).to(dev)
        optimizer = torch.optim.Adam(gate.parameters(), lr=float(gcfg.lr), weight_decay=float(gcfg.weight_decay))
        f_tr, s_tr = feats["val"][gate_train].to(dev), structs["val"][gate_train].to(dev)
        p_tr, y_tr = probs["val"][:, gate_train].to(dev), labels["val"][gate_train].to(dev)
        f_va, s_va = feats["val"][gate_val].to(dev), structs["val"][gate_val].to(dev)
        p_va, y_va = probs["val"][:, gate_val].to(dev), labels["val"][gate_val]

        best_auc, best, last = float("-inf"), None, None
        epochs = int(gcfg.epochs)
        for epoch in range(1, epochs + 1):
            gate.train()
            loss = gate_loss(mixture_probability(gate(f_tr, s_tr), p_tr), y_tr, float(gcfg.neg_loss_weight))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gate.eval()
            with torch.no_grad():
                val_metrics = probability_metrics(mixture_probability(gate(f_va, s_va), p_va).cpu(), y_va, prefix="val_")
            self.train_history.append({"epoch": epoch, "loss": float(loss.item()), "metrics": val_metrics})
            last = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_metrics": val_metrics,
                "state": {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
            }
            auc = val_metrics.get("val_auc", float("nan"))
            if auc > best_auc:  # NaN never improves
                best_auc, best = auc, last
            if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
                print(f"[LinkMoE][Gate {epoch}/{epochs}] train_loss={loss.item():.4f} val_auc={auc:.4f} best_auc={best_auc:.4f}")
        best = best or last  # gate-val AUC undefined throughout: keep the last epoch
        gate.load_state_dict(best["state"])
        return gate, optimizer, best

    def _gate_diagnostics(self, weights: torch.Tensor, structs, probs, labels) -> dict:
        cn = structs["test"][:, HEURISTIC_NAMES.index("cn")]
        by_cn = {}
        for tag, lo, hi in _CN_BUCKETS:
            mask = (cn >= lo) & (cn < hi)
            if mask.any():
                by_cn[tag] = dict(zip(self.expert_names, weights[mask].mean(0).tolist()))
        return {
            "mean_ensemble_test_auc": probability_metrics(probs["test"].mean(0), labels["test"])["auc"],
            "gate_weight_mean_test": dict(zip(self.expert_names, weights.mean(0).tolist())),
            "gate_weight_by_cn_test": by_cn,
        }

    def fit(self) -> None:
        if self._skip_due_to_existing_checkpoint:
            return
        seed = int(self.cfg.seed)
        views = build_link_views(self.cfg, seed)
        self.dataset_meta = dict(views.meta)
        print(
            f"[LinkMoE] {self.lcfg.dataset.name}: nodes={views.num_nodes} "
            + " ".join(f"{s}_pairs={views.pairs[s].size(1)}" for s in ("train", "val", "test"))
        )

        scores = self._train_experts(views, seed)
        probs = {s: torch.stack([getattr(sc, f"{s}_prob") for sc in scores]) for s in ("val", "test")}
        feats, structs = self._gate_inputs(views)
        gate_train, gate_val = stratified_split(views.labels["val"], float(self.lcfg.gate.val_train_ratio), seed)
        gate, optimizer, best = self._train_gate(feats, structs, probs, views.labels, gate_train, gate_val, seed)

        gate.eval()
        with torch.no_grad():
            weights = gate(feats["test"].to(self.device), structs["test"].to(self.device)).cpu()
        q_test = mixture_probability(weights, probs["test"])
        test_metrics = probability_metrics(q_test, views.labels["test"], prefix="test_")
        self.best_epoch = int(best["epoch"])
        self.best_metrics = {
            "train_loss": best["train_loss"],
            "best_epoch": self.best_epoch,
            **best["val_metrics"],
            **test_metrics,
        }
        self._save(gate, optimizer, scores, gate_train, gate_val, self._gate_diagnostics(weights, structs, probs, views.labels))
        print(
            f"[LinkMoE] Complete. best_epoch={self.best_epoch} "
            f"val_auc={self.best_metrics.get('val_auc', float('nan')):.4f} "
            f"test_auc={self.best_metrics.get('test_auc', float('nan')):.4f}"
        )

    def _save(self, gate, optimizer, scores, gate_train, gate_val, diagnostics: dict) -> None:
        save_checkpoint(
            path=self._checkpoint_path(),
            model=gate,
            optimizer=optimizer,
            epoch=self.best_epoch,
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=self.best_metrics,
            extra={
                "experts": list(self.expert_names),
                "expert_states": {sc.name: sc.state for sc in scores},
                "expert_scores": {sc.name: {"val_prob": sc.val_prob, "test_prob": sc.test_prob} for sc in scores},
                "heuristic_names": list(HEURISTIC_NAMES),
                "gate_split": {"train": gate_train, "val": gate_val},
            },
        )
        save_training_log(
            path=self._log_path(),
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            history=self.train_history,
            best_info={"epoch": self.best_epoch, "metric": self.best_metrics.get("val_auc"), "monitor": "val_auc"},
            extra={
                "experts": {
                    sc.name: {"val_auc": sc.val_auc, "test_auc": sc.test_auc, "best_epoch": sc.best_epoch}
                    for sc in scores
                },
                **diagnostics,
            },
        )
        print(f"[LinkMoE] Saved checkpoint: {self._checkpoint_path()}")


__all__ = ["LinkMoERunner", "validate_link_task"]
