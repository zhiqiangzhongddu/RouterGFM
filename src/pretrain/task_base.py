from torch import nn


class PretrainTask(nn.Module):
    """Base class for pretraining objectives.

    Task classes may own auxiliary trainable modules (e.g. a projection
    head, discriminator, or secondary encoder). Those modules are picked
    up by ``parameters_to_optimize()`` and trained jointly with the main
    encoder. ``PretrainRunner`` persists both the main encoder and the task
    state for the selected checkpoint so supervised heads and contrastive
    auxiliaries can be restored for final evaluation. Encoder-only warm starts
    continue to read just the checkpoint's ``model_state``.
    """

    name = "base"

    #: Default monitor metric for this task.  When set (e.g.
    #: ``"train_loss"``), the monitoring resolver uses this value
    #: instead of auto-resolving from task_type/task_level.  ``None``
    #: means "use the auto policy".  This replaces method-name string
    #: checks in the per-workflow monitoring modules.
    default_monitor: str | None = None

    #: When True, the pretrain runner must feed the task graph-level
    #: batches. Node/edge datasets are promoted via induced subgraphs;
    #: a method declaring this flag rejects raw node/edge datasets with
    #: ``induced=False``.
    requires_graph_batches: bool = False

    #: .. deprecated::
    #:    Superseded by ``min_graphs_per_batch``.  The runner now
    #:    derives ``drop_last`` solely from ``min_graphs_per_batch > 1``.
    #:    This attribute is kept for backward compatibility but is no
    #:    longer consulted by the runner.  New methods should only set
    #:    ``min_graphs_per_batch``.
    drop_last_singleton_batch: bool = False

    #: Minimum number of graphs a ``step()`` call needs to produce a
    #: real loss. Methods that build a cross-graph similarity or sampling
    #: structure (GraphCL, InfoGraph, ContextPred) set this to ``2``.
    #: The runner keeps partial batches so valid non-full batches are not
    #: discarded. If a too-small tail reaches ``step()``, the method is
    #: responsible for returning an anchored zero-loss via ``make_zero_loss``
    #: instead of hard-raising.
    min_graphs_per_batch: int = 1

    #: When True, the task participates in the dataset's train/val/test
    #: split and the runner builds split loaders + calls ``evaluate()``
    #: for val/test metrics.
    uses_dataset_splits: bool = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        """Validate method-specific config options before training starts.

        Called by ``build_pretrain_task`` right before instantiation.
        Must raise ``ValueError`` with a clear message on bad input.
        Default is a no-op; methods override to fail fast on invalid
        enums / ratios / pool names instead of crashing mid-training.

        Intentionally not called from ``variant_tag_for``: run-name
        construction is also used by non-pretraining code paths (e.g.
        finetune checkpoint resolution) that only need the derived
        filename and should not fail if the pretrain block carries
        options that would be invalid in a real pretraining run.
        """
        return None

    def parameters_to_optimize(self):
        """Return task parameters that should be optimized during pretraining."""
        return self.parameters()

    @classmethod
    def run_tag(cls, cfg) -> str:
        """Return the primary method identity tag for the run name.

        This is the coarser tag (e.g. ``"plus1"``), whereas
        :meth:`variant_tag` captures secondary option deviations.
        Default returns ``""``; methods override when needed.
        """
        return ""

    @classmethod
    def variant_tag(cls, cfg) -> str:
        """Return a short filename-safe tag describing method-variant options.

        The tag is appended to the pretrain run name so that two runs that
        differ only in method-specific knobs (e.g. ``context_pred.mode``,
        ``edge_pred.use_mlp_scorer``, ``graphcl.aug1/aug2``) produce
        distinct checkpoints and logs.

        Convention: emit tag components **only** for fields that differ
        from the default declared in ``src/config.py``. A run that uses
        all-default variant options must return ``""`` so its filename is
        unchanged from the pre-variant-tag era.
        """
        return ""

    def step(self, model: nn.Module, data, device):
        """Run one training step and return ``(loss, log_dict)``.

        Log-dict key convention (enforced by ``PretrainRunner``):
          - Scalar metrics (e.g. ``train_acc``, ``pos_mean``, ``sim``)
            are averaged across batches and printed each epoch.
          - Bookkeeping counts/sizes (e.g. ``masked_count``,
            ``num_pairs_count``, ``batch_size_count``) end with the
            ``_count`` suffix. The runner retains them in the history
            but skips them in per-epoch stdout, so counters can be
            added without editing the runner.
        """
        raise NotImplementedError

    def evaluate(
        self,
        model: nn.Module,
        data,
        device,
        mask_attr: str = "val_mask",
        return_outputs: bool = False,
    ):
        """Optional evaluation hook for supervised-style pretraining tasks.

        Unsupervised tasks may leave this unimplemented; only the pretrain
        runner's supervised evaluation path calls into it.
        """
        raise NotImplementedError(
            "evaluate() is only implemented for supervised-style pretraining tasks."
        )
