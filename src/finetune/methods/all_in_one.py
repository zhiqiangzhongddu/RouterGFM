"""All-in-One prompt finetuning method."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.finetune.prompts.all_in_one import HeavyPrompt
from src.finetune.registry import register
from src.finetune.task_heads import TaskAwareObjective, align_last_dim, build_task_aware_classifier, prepare_single_label_labels
from src.finetune.task_base import FinetuneTask
from src.utils.pool import get_batch_vector, pool_nodes
from src.utils.supervised_eval import evaluate_epoch_split
from src.utils.config_helpers import cfg_default, resolve_method_optim_field, tag_if_nondefault
from src.utils.dataset_helpers import is_few_shot_split, read_effective_task_level
from src.utils.parsing import resolve_task_type, to_bool


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def _none_if_str_none(value):
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"none", "null", ""}:
            return None
    return value


@register("all_in_one")
class FinetuneAllInOne(FinetuneTask):
    """Alternating prompt/answer optimization on graph-level batches."""

    requires_frozen_encoder = True
    supports_early_stopping = False
    default_monitor = "train_loss"

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        cls.require_graph_level_batches(cfg, method_label="All-in-One")

        aio_cfg = getattr(getattr(cfg, "finetune", None), "all_in_one", None)
        if aio_cfg is None:
            return
        raw_token = getattr(aio_cfg, "token_num", 10)
        token_num = int(raw_token) if raw_token is not None else 10
        if token_num < 1:
            raise ValueError(f"[All-in-One] token_num must be >= 1, got {token_num}")
        raw_cp = getattr(aio_cfg, "cross_prune", 0.1)
        cross_prune = float(raw_cp) if raw_cp is not None else 0.1
        raw_ip = getattr(aio_cfg, "inner_prune", 0.3)
        inner_prune = float(raw_ip) if raw_ip is not None else 0.3
        if not (0.0 <= cross_prune <= 1.0):
            raise ValueError(f"[All-in-One] cross_prune must be in [0, 1], got {cross_prune}")
        if not (0.0 <= inner_prune <= 1.0):
            raise ValueError(f"[All-in-One] inner_prune must be in [0, 1], got {inner_prune}")

    @classmethod
    def run_tag(cls, cfg) -> str:
        aio_cfg = getattr(getattr(cfg, "finetune", None), "all_in_one", None)
        raw_token = getattr(aio_cfg, "token_num", 10) if aio_cfg is not None else 10
        token_num = int(raw_token) if raw_token is not None else 10
        return f"tk{token_num}"

    @classmethod
    def variant_tag(cls, cfg) -> str:
        aio_cfg = getattr(getattr(cfg, "finetune", None), "all_in_one", None)
        if aio_cfg is None:
            return ""
        tags = []
        raw_cp = getattr(aio_cfg, "cross_prune", 0.1)
        cross_prune = float(raw_cp) if raw_cp is not None else 0.1
        t = tag_if_nondefault("cp", cross_prune, cfg_default("finetune.all_in_one.cross_prune"))
        if t:
            tags.append(t)
        raw_ip = getattr(aio_cfg, "inner_prune", 0.3)
        inner_prune = float(raw_ip) if raw_ip is not None else 0.3
        t = tag_if_nondefault("ip", inner_prune, cfg_default("finetune.all_in_one.inner_prune"))
        if t:
            tags.append(t)
        raw_te = getattr(aio_cfg, "total_epochs", 1000)
        total_epochs = int(raw_te) if raw_te is not None else 1000
        t = tag_if_nondefault("te", total_epochs, cfg_default("finetune.all_in_one.total_epochs"))
        if t:
            tags.append(t)
        answer_epoch = int(getattr(aio_cfg, "answer_epoch", -1))
        if answer_epoch > 0:
            tags.append(f"ae{answer_epoch}")
        prompt_epoch = int(getattr(aio_cfg, "prompt_epoch", -1))
        if prompt_epoch > 0:
            tags.append(f"pe{prompt_epoch}")
        cache = bool(getattr(aio_cfg, "cache_answer_embeddings", True))
        if not cache:
            tags.append("nocache")
        softmax = bool(getattr(aio_cfg, "answer_with_softmax", False))
        if softmax:
            tags.append("softmax")
        bidir = getattr(aio_cfg, "bidirectional_cross_edges", None)
        if bidir is not None:
            tags.append(f"bidir{int(bool(bidir))}")
        exclude = getattr(aio_cfg, "exclude_prompt_from_pooling", None)
        if exclude is not None:
            tags.append(f"excl{int(bool(exclude))}")
        return "-".join(tags)

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = read_effective_task_level(ds_cfg)
        self.task_level_raw = str(ds_cfg.task_level or self.task_level).lower()
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)

        method_cfg = getattr(cfg.finetune, "all_in_one", None)
        token_num = int(getattr(method_cfg, "token_num", 10)) if method_cfg else 10
        cross_prune = float(getattr(method_cfg, "cross_prune", 0.1)) if method_cfg else 0.1
        inner_prune = float(getattr(method_cfg, "inner_prune", 0.3)) if method_cfg else 0.3
        # Reference downstream scripts use epochs=1000 by default for
        # All-in-One. Keep this method-specific so other finetune methods can
        # use different global epoch budgets.
        self.total_epochs = int(getattr(method_cfg, "total_epochs", 1000)) if method_cfg else 1000
        # Speed-up: answer phase has frozen encoder + frozen prompt, so cache
        # graph embeddings once per outer epoch and optimize the head on cache.
        self.cache_answer_embeddings = (
            to_bool(getattr(method_cfg, "cache_answer_embeddings", True)) if method_cfg else True
        )
        # Strict parity mode with the reference All-in-One head uses
        # Linear + Softmax as the answering head while still optimizing with CE.
        self.answer_with_softmax = (
            to_bool(getattr(method_cfg, "answer_with_softmax", False)) if method_cfg else False
        )
        # Historical best runs in this repo consistently used bidirectional
        # prompt-graph links and excluded prompt nodes from readout on induced
        # node tasks, while reference defaults are one-way links with
        # prompt-inclusive pooling.
        default_bidirectional = self.task_level_raw == "node"
        default_exclude_prompt = self.task_level_raw == "node"
        bidirectional_cross_edges_cfg = _none_if_str_none(
            getattr(method_cfg, "bidirectional_cross_edges", None) if method_cfg else None
        )
        exclude_prompt_from_pooling_cfg = _none_if_str_none(
            getattr(method_cfg, "exclude_prompt_from_pooling", None) if method_cfg else None
        )
        self.bidirectional_cross_edges = (
            to_bool(bidirectional_cross_edges_cfg)
            if bidirectional_cross_edges_cfg is not None
            else default_bidirectional
        )
        self.exclude_prompt_from_pooling = (
            to_bool(exclude_prompt_from_pooling_cfg)
            if exclude_prompt_from_pooling_cfg is not None
            else default_exclude_prompt
        )
        is_few_shot = is_few_shot_split(getattr(ds_cfg, "fixed_split", None))
        # Reference schedule differs by task path:
        # - node_task and graph few-shot: answer/prompt = 50/50
        # - graph-task standard split: answer/prompt = 5/1
        if self.task_level_raw == "node":
            default_answer_epoch = 50
            default_prompt_epoch = 50
        else:
            default_answer_epoch = 50 if is_few_shot else 5
            default_prompt_epoch = 50 if is_few_shot else 1
        answer_epoch_cfg = int(getattr(method_cfg, "answer_epoch", -1)) if method_cfg else -1
        prompt_epoch_cfg = int(getattr(method_cfg, "prompt_epoch", -1)) if method_cfg else -1
        self.answer_epoch = max(1, answer_epoch_cfg if answer_epoch_cfg > 0 else default_answer_epoch)
        self.prompt_epoch = max(1, prompt_epoch_cfg if prompt_epoch_cfg > 0 else default_prompt_epoch)
        # Reference monitored loss differs by task path:
        # - node_task.AllInOneTrain returns answer_loss
        # - graph_task.AllInOneTrain returns pg_loss (prompt_loss)
        self._monitor_prompt_loss = self.task_level_raw != "node"

        num_classes = int(getattr(ds_cfg, "num_classes", 2) or 2)
        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        out_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.repr_dim = out_dim
        self.objective = TaskAwareObjective(cfg, task_level=self.task_level, repr_dim=self.repr_dim)
        self.single_label_classification = self.objective.is_single_label_classification

        self.prompt = HeavyPrompt(
            token_dim=int(cfg.model.in_dim),
            token_num=token_num,
            cross_prune=cross_prune,
            inner_prune=inner_prune,
            bidirectional_cross_edges=self.bidirectional_cross_edges,
        )
        if self.answer_with_softmax and not self.single_label_classification:
            raise ValueError("All-in-One answer_with_softmax is only supported for single-label classification.")
        if self.answer_with_softmax:
            self.answering = nn.Sequential(
                nn.Linear(self.repr_dim, self.objective.output_dim),
                nn.Softmax(dim=1),
            )
        else:
            self.answering = build_task_aware_classifier(
                input_dim=self.repr_dim,
                task_type=self.task_type,
                label_dim=self.label_dim,
                num_classes=num_classes,
            )
        self._answer_cache_notice_printed = False


    def get_effective_epochs(self, original_epochs: int) -> int:
        total_epochs = int(self.total_epochs) if int(self.total_epochs) > 0 else int(original_epochs)
        if self.answer_epoch > 0:
            effective = total_epochs // self.answer_epoch
            return max(1, effective)
        return total_epochs

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.answering.parameters())

    def build_optimizers(self, model: nn.Module):
        method_cfg = getattr(self.cfg.finetune, "all_in_one", None)
        prompt_lr = resolve_method_optim_field(method_cfg, "prompt_lr", 1e-6)
        prompt_wd = resolve_method_optim_field(method_cfg, "prompt_weight_decay", float(self.cfg.finetune.weight_decay))
        answer_lr = resolve_method_optim_field(method_cfg, "answer_lr", float(self.cfg.finetune.lr))
        answer_wd = resolve_method_optim_field(method_cfg, "answer_weight_decay", float(self.cfg.finetune.weight_decay))

        prompt_opt = torch.optim.Adam(self.prompt.parameters(), lr=prompt_lr, weight_decay=prompt_wd)
        answer_opt = torch.optim.Adam(self.answering.parameters(), lr=answer_lr, weight_decay=answer_wd)
        return {"prompt": prompt_opt, "answer": answer_opt, "primary": answer_opt}

    def _graph_embeddings(self, model: nn.Module, batch, device):
        batch = batch.to(device)
        prompted = self.prompt(batch)
        node_repr, graph_repr = model(prompted)
        pool_mode = str(getattr(self.cfg.model, "graph_pooling", "mean") or "mean").lower()
        if pool_mode == "sum":
            pool_mode = "add"
        batch_vec = get_batch_vector(prompted)

        if self.exclude_prompt_from_pooling and hasattr(prompted, "prompt_node_mask"):
            prompt_mask = torch.as_tensor(prompted.prompt_node_mask, dtype=torch.bool, device=node_repr.device)
            node_mask = ~prompt_mask
            if bool(node_mask.any().item()):
                graph_repr = pool_nodes(
                    x=node_repr[node_mask],
                    batch=batch_vec[node_mask],
                    mode=pool_mode,
                )
            elif graph_repr is None:
                graph_repr = pool_nodes(
                    x=node_repr,
                    batch=batch_vec,
                    mode=pool_mode,
                )
        elif graph_repr is None:
            graph_repr = pool_nodes(
                x=node_repr,
                batch=batch_vec,
                mode=pool_mode,
            )
        graph_repr = align_last_dim(graph_repr, self.repr_dim)
        labels = torch.as_tensor(prompted.y).to(device)
        return graph_repr, labels

    def _answer_forward(self, graph_repr, labels, return_outputs: bool = False):
        graph_repr = align_last_dim(graph_repr, self.repr_dim)
        if self.answer_with_softmax:
            logits = self.answering(graph_repr)
            class_labels = prepare_single_label_labels(labels).to(graph_repr.device)
            # F.cross_entropy applies log-softmax internally, producing
            # log(softmax(softmax(x))). This replicates a known bug in the
            # reference  implementation for strict parity.
            loss = F.cross_entropy(logits, class_labels)
            pred = logits.argmax(dim=-1)
            acc = float((pred == class_labels).float().mean().item())
            if return_outputs:
                return loss, acc, torch.log(logits.clamp_min(1e-12)), class_labels
            return loss, acc
        return self.objective.forward_with_classifier(
            classifier=self.answering,
            representations=graph_repr,
            labels=labels,
            input_dim=self.repr_dim,
            return_outputs=return_outputs,
        )

    def _run_epoch(self, model, loader, device, optimizer, train_prompt: bool):
        if train_prompt:
            self.prompt.train()
            self.answering.eval()
            _set_requires_grad(self.prompt, True)
            _set_requires_grad(self.answering, False)
        else:
            self.prompt.eval()
            self.answering.train()
            _set_requires_grad(self.prompt, False)
            _set_requires_grad(self.answering, True)

        total_loss = 0.0
        num_batches = 0
        for data in loader:
            optimizer.zero_grad(set_to_none=True)
            if train_prompt:
                graph_repr, labels = self._graph_embeddings(model, data, device)
            else:
                with torch.no_grad():
                    graph_repr, labels = self._graph_embeddings(model, data, device)
            loss, _primary = self._answer_forward(graph_repr, labels, return_outputs=False)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Train loader is empty; unable to run a training epoch.")
        return total_loss / num_batches

    def _build_answer_cache(self, model, loader, device):
        self.prompt.eval()
        _set_requires_grad(self.prompt, False)
        _set_requires_grad(self.answering, True)

        cache = []
        with torch.no_grad():
            for data in loader:
                graph_repr, labels = self._graph_embeddings(model, data, device)
                cache.append((graph_repr.detach(), labels.detach()))
        return cache

    def _run_answer_epoch_from_cache(self, cache, optimizer):
        if not cache:
            return 0.0

        self.answering.train()
        self.prompt.eval()
        _set_requires_grad(self.prompt, False)
        _set_requires_grad(self.answering, True)

        if len(cache) > 1:
            order = torch.randperm(len(cache), device=cache[0][0].device).tolist()
        else:
            order = [0]

        total_loss = 0.0
        for idx in order:
            graph_repr, labels = cache[idx]
            optimizer.zero_grad(set_to_none=True)
            loss, _primary = self._answer_forward(graph_repr, labels, return_outputs=False)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return total_loss / len(cache)

    def train_epoch(self, model, loader, device, optimizers=None):
        if not isinstance(optimizers, dict):
            raise ValueError("All-in-One requires named optimizers.")

        answer_opt = optimizers.get("answer")
        prompt_opt = optimizers.get("prompt")
        if answer_opt is None or prompt_opt is None:
            raise ValueError("All-in-One optimizers are missing.")

        answer_loss = 0.0
        prompt_loss = 0.0
        if self.cache_answer_embeddings and self.answer_epoch > 0:
            if not self._answer_cache_notice_printed:
                print("[Finetune][all_in_one] Enabled cached answering-phase embeddings.")
                self._answer_cache_notice_printed = True
            answer_cache = self._build_answer_cache(model, loader, device)
            for _ in range(self.answer_epoch):
                answer_loss = self._run_answer_epoch_from_cache(answer_cache, answer_opt)
        else:
            for _ in range(self.answer_epoch):
                answer_loss = self._run_epoch(model, loader, device, answer_opt, train_prompt=False)
        for _ in range(self.prompt_epoch):
            prompt_loss = self._run_epoch(model, loader, device, prompt_opt, train_prompt=True)

        monitored_loss = prompt_loss if self._monitor_prompt_loss else answer_loss
        return monitored_loss, {
            "train_answer_loss": float(answer_loss),
            "train_prompt_loss": float(prompt_loss),
        }

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> dict[str, float]:
        del mask_attr  # graph-level evaluation does not use masks

        model.eval()
        self.prompt.eval()
        self.answering.eval()

        def _forward(data, device):
            graph_repr, labels = self._graph_embeddings(model, data, device)
            loss, _primary, logits, labels = self._answer_forward(
                graph_repr, labels, return_outputs=True,
            )
            return loss, logits, labels

        return evaluate_epoch_split(
            forward_fn=_forward,
            loader=loader,
            device=device,
            prefix=prefix,
            task_type=self.task_type,
        )
