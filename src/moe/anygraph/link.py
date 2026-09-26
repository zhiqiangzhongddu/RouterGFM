"""Code-owned wrapper around the embedded AnyGraph link runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Optional, Sequence

from src.utils import parse_csv_list, resolve_project_path

from .runtime import (
    ANYGRAPH_HISTORY_DIR,
    ANYGRAPH_LINK_MAIN,
    ANYGRAPH_MODELS_DIR,
    ANYGRAPH_SRC_ROOT,
    REPO_ROOT,
    build_anygraph_runtime_env,
    ensure_anygraph_runtime_files_exist,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "anygraph_data" / "link"
DEFAULT_RESULT_CSV = REPO_ROOT / "outputs" / "anygraph" / "anygraph_link_eval.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AnyGraph link route.")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names.")
    parser.add_argument("--dataset_setting", default="", help="Explicit dataset_setting override.")
    parser.add_argument("--result_csv", default=str(DEFAULT_RESULT_CSV))
    parser.add_argument("--save_path", default="anygraph_link_run")
    parser.add_argument("--load_model", default="")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epoch", type=int, default=0, help="Set 0 for eval-only with --load_model.")
    parser.add_argument("--tst_epoch", type=int, default=1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--eval_protocol", default="agae", choices=["agae"])
    parser.add_argument("--edge_eval_threshold_mode", default="val_best_acc", choices=["val_best_acc", "zero"])
    parser.add_argument("--edge_eval_payload_name", default="agae_edge_eval_payload.pt")
    parser.add_argument("--edge_eval_repeat_times", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, extra = parser.parse_known_args(argv)

    ensure_anygraph_runtime_files_exist(ANYGRAPH_LINK_MAIN)

    data_root = resolve_project_path(args.data_root)
    result_csv = resolve_project_path(args.result_csv)
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    ANYGRAPH_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ANYGRAPH_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv_list(args.datasets)
    if not datasets and not args.dataset_setting:
        raise ValueError(
            "Strict mode requires explicit datasets. "
            "Provide --datasets or --dataset_setting."
        )

    dataset_setting = args.dataset_setting if args.dataset_setting else ",".join(datasets)
    cmd = [
        sys.executable,
        "main.py",
        "--data_root",
        str(data_root),
        "--dataset_setting",
        dataset_setting,
        "--result_csv",
        str(result_csv),
        "--save_path",
        str(args.save_path),
        "--gpu",
        str(args.gpu),
        "--epoch",
        str(int(args.epoch)),
        "--tst_epoch",
        str(int(args.tst_epoch)),
        "--topk",
        str(int(args.topk)),
        "--eval_protocol",
        str(args.eval_protocol),
        "--edge_eval_threshold_mode",
        str(args.edge_eval_threshold_mode),
        "--edge_eval_payload_name",
        str(args.edge_eval_payload_name),
        "--edge_eval_repeat_times",
        str(int(args.edge_eval_repeat_times)),
    ]
    if args.load_model:
        cmd.extend(["--load_model", str(args.load_model)])
    if extra:
        cmd.extend(str(item) for item in extra)

    print(f"[MoE][AnyGraph][Link] cwd={ANYGRAPH_SRC_ROOT}")
    print("[MoE][AnyGraph][Link] cmd=", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(ANYGRAPH_SRC_ROOT),
        check=False,
        env=build_anygraph_runtime_env(),
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
