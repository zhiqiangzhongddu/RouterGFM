"""CLI entrypoint for mixture-of-experts (MoE) methods.

Select a method via ``moe.method`` (dispatch table: ``src/moe/run.py``):

- ``anygraph`` — the AnyGraph pipeline (stage via ``moe.anygraph.execution.step``).
- ``routergfm`` — the RouterGFM router-based GFM pipeline (stage via ``moe.routergfm.task``).
- ``gmoe`` — the Graph Mixture of Experts encoder, trained end-to-end (config under ``moe.gmoe``).
- ``mowst`` — Mowst weak/strong per-node expert mixture (variant via ``moe.mowst.variant``).
- ``graphmore`` — Mixture of Riemannian Experts (config under ``moe.graphmore``).
- ``gmope`` — Graph Mixture of Prompt-Experts (stage via ``moe.gmope.stage``).
- ``nodemoe`` — Node-MoE node-wise filtering experts, node tasks only.
- ``linkmoe`` — Link-MoE mixture of link predictors, link tasks only.
- ``graphmetro`` / ``ogmm`` / ``geomoe`` — shift-robust mixtures; a shift condition
  is read from ``data_preparation.dataset.split_root``.

Every method except ``anygraph`` and ``routergfm`` also runs a whole sweep with
``moe.<method>.run_tasks_tsv True moe.<method>.tasks_tsv slurm/moe.<method>.all.tsv``.

Examples:
    python scripts/run_moe.py moe.method anygraph moe.anygraph.execution.step train \\
      moe.anygraph.train.dataset.name cora

    python scripts/run_moe.py moe.method gmoe moe.gmoe.dataset.name cora \\
      moe.gmoe.dataset.task_level node moe.gmoe.dataset.induced True

    python scripts/run_moe.py moe.method mowst moe.mowst.variant mowst_star \\
      moe.mowst.dataset.name chameleon moe.mowst.dataset.fixed_split "(5,0.0,1.0)"

    python scripts/run_moe.py moe.method graphmore moe.graphmore.dataset.name mnist \\
      moe.graphmore.dataset.task_level graph moe.graphmore.dataset.fixed_split "(5,0.0,1.0)"

    python scripts/run_moe.py moe.method gmope moe.gmope.stage all moe.gmope.dataset.name photo \\
      moe.gmope.dataset.fixed_split "(5,0.0,1.0)"

    python scripts/run_moe.py moe.method nodemoe moe.nodemoe.dataset.name chameleon \\
      moe.nodemoe.dataset.fixed_split "(5,0.0,1.0)"

    python scripts/run_moe.py moe.method linkmoe moe.linkmoe.dataset.name dblp \\
      moe.linkmoe.dataset.fixed_split "(0.1,0.05,0.1)"

    python scripts/run_moe.py moe.method graphmetro moe.graphmetro.dataset.name photo \\
      moe.graphmetro.dataset.fixed_split "(5,0.0,1.0)"

    python scripts/run_moe.py moe.method ogmm moe.ogmm.dataset.name airports \\
      moe.ogmm.dataset.fixed_split "(5,0.0,1.0)" data_preparation.dataset.split_root data/splits_shift/feature

    python scripts/run_moe.py moe.method geomoe moe.geomoe.dataset.name mnist \\
      moe.geomoe.dataset.task_level graph data_preparation.dataset.split_root data/splits_shift/structural

RouterGFM stages (``moe.routergfm.task``; see ``src/moe/routergfm/run.py``):

    # history: diagnostic records of one expert shard (array job over shards)
    python scripts/run_moe.py moe.method routergfm moe.routergfm.task history \\
      moe.routergfm.experts.shard_index 0 moe.routergfm.experts.num_shards 48

    # router: train the router of one (target, budget) task
    python scripts/run_moe.py moe.method routergfm moe.routergfm.task router \\
      moe.routergfm.deploy.target photo:node moe.routergfm.deploy.budget 5

    # deploy: one target application (dataset:level, budget, split seed)
    python scripts/run_moe.py moe.method routergfm moe.routergfm.task deploy \\
      moe.routergfm.deploy.target photo:node moe.routergfm.deploy.budget 5 moe.routergfm.deploy.seed 42

    # benchmark: every TSV row over apps.seeds[:num_runs] -> results table
    python scripts/run_moe.py moe.method routergfm moe.routergfm.task benchmark \\
      moe.routergfm.benchmark.run_tasks_tsv True moe.routergfm.benchmark.tasks_tsv slurm/moe.routergfm.tsv

    # analysis: one App. D diagnostic on the deploy target
    python scripts/run_moe.py moe.method routergfm moe.routergfm.task analysis \\
      moe.routergfm.analysis.kind team_size moe.routergfm.deploy.target photo:node

    # selection_baseline / matched_baseline: over baselines.datasets x baselines.budgets
    python scripts/run_moe.py moe.method routergfm moe.routergfm.task selection_baseline \\
      moe.routergfm.baselines.method logme
    python scripts/run_moe.py moe.method routergfm moe.routergfm.task matched_baseline \\
      moe.routergfm.baselines.method sagmm_pe
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable, Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.moe.run import run_moe_from_cli


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return run_moe_from_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
