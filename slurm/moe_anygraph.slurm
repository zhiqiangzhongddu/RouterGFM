#!/bin/bash -l
#SBATCH --job-name "anygraph"
#SBATCH --time=24:00:00
#SBATCH --account=p201211
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=default
#SBATCH --output=slurm/output/%x-%j.out
#SBATCH --error=slurm/output/%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"

export PYTHONUNBUFFERED=1

STEP="${STEP:-all}"   # conversion | train | eval | all
case "${STEP}" in
  conversion|train|eval|all) ;;
  *)
    echo "[AnyGraph] Invalid STEP=${STEP}. Use conversion, train, eval, or all."
    exit 1
    ;;
esac

GPU_ID="${SLURM_LOCALID:-${SLURM_PROCID:-0}}"
DEVICE="${DEVICE:-${GPU_ID}}"

DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/data/datasets}"
SPLIT_ROOT="${SPLIT_ROOT:-${ROOT_DIR}/data/splits}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/data/anygraph_data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${ROOT_DIR}/outputs/anygraph}"
mkdir -p "${OUTPUTS_DIR}"

# Shared stage defaults. Stage-specific env vars override these.
MODE="${MODE:-all}"    # all | node | link | both | graph
DATASET="${DATASET:-}"
FIXED_SPLIT="${FIXED_SPLIT:-}"
LINK_DATASET_SETTING="${LINK_DATASET_SETTING:-}"
NODE_DATASET_SETTING="${NODE_DATASET_SETTING:-}"
GRAPH_DATASET_SETTING="${GRAPH_DATASET_SETTING:-}"
MIXED_DATASET_SETTING="${MIXED_DATASET_SETTING:-}"
LOAD_LINK_MODEL="${LOAD_LINK_MODEL:-}"
LOAD_NODE_MODEL="${LOAD_NODE_MODEL:-}"
LOAD_GRAPH_MODEL="${LOAD_GRAPH_MODEL:-}"
SAVE_LINK_PATH="${SAVE_LINK_PATH:-}"   # empty = auto run-name from cfg (datasets/seeds/epoch)
SAVE_NODE_PATH="${SAVE_NODE_PATH:-}"   # empty = auto run-name from cfg (datasets/seeds/epoch)
SAVE_GRAPH_PATH="${SAVE_GRAPH_PATH:-}" # empty = auto run-name from cfg (datasets/seeds/epoch)

# Conversion options.
TASK="${TASK:-auto}"                    # auto | all | node | link
SEED="${SEED:-42}"
MASK_COL="${MASK_COL:-0}"
FEAT_REDUCTION="${FEAT_REDUCTION:-0}"   # 0/1
L1_NORMALIZE_FEATURES="${L1_NORMALIZE_FEATURES:-1}"  # 0/1
EMIT_NODE_VAL="${EMIT_NODE_VAL:-1}"     # 0/1
FEAT_DIM="${FEAT_DIM:-100}"
EDGE_EVAL_PAYLOAD_NAME="${EDGE_EVAL_PAYLOAD_NAME:-agae_edge_eval_payload.pt}"
NODE_OUTPUT_FEAT_DIM="${NODE_OUTPUT_FEAT_DIM:-128}"
DATASET_FILE="${DATASET_FILE:-${ROOT_DIR}/data/available_node_datasets.tsv}"
SEEDS="${SEEDS:-${SEED}}"
EDGE_SPLITS="${EDGE_SPLITS:-}"
NODE_SPLITS="${NODE_SPLITS:-}"
GRAPH_SPLITS="${GRAPH_SPLITS:-}"

# Train options.
TRAIN_MODE="${TRAIN_MODE:-${MODE}}"
TRAIN_EPOCH="${TRAIN_EPOCH:-${EPOCH:-100}}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"   # 0/1
TRAIN_SAVE_LINK_PATH="${TRAIN_SAVE_LINK_PATH:-${SAVE_LINK_PATH}}"
TRAIN_SAVE_NODE_PATH="${TRAIN_SAVE_NODE_PATH:-${SAVE_NODE_PATH}}"
TRAIN_SAVE_GRAPH_PATH="${TRAIN_SAVE_GRAPH_PATH:-${SAVE_GRAPH_PATH}}"
TRAIN_LOAD_LINK_MODEL="${TRAIN_LOAD_LINK_MODEL:-${LOAD_LINK_MODEL}}"
TRAIN_LOAD_NODE_MODEL="${TRAIN_LOAD_NODE_MODEL:-${LOAD_NODE_MODEL}}"
TRAIN_LOAD_GRAPH_MODEL="${TRAIN_LOAD_GRAPH_MODEL:-${LOAD_GRAPH_MODEL}}"
TRAIN_DATASET="${TRAIN_DATASET:-${DATASET}}"
TRAIN_FIXED_SPLIT="${TRAIN_FIXED_SPLIT:-${FIXED_SPLIT}}"
TRAIN_LINK_DATASET_SETTING="${TRAIN_LINK_DATASET_SETTING:-${LINK_DATASET_SETTING}}"
TRAIN_NODE_DATASET_SETTING="${TRAIN_NODE_DATASET_SETTING:-${NODE_DATASET_SETTING}}"
TRAIN_GRAPH_DATASET_SETTING="${TRAIN_GRAPH_DATASET_SETTING:-${GRAPH_DATASET_SETTING}}"
TRAIN_MIXED_DATASET_SETTING="${TRAIN_MIXED_DATASET_SETTING:-${MIXED_DATASET_SETTING}}"

# Eval options.
EVAL_MODE="${EVAL_MODE:-${MODE}}"
EVAL_SAVE_LINK_PATH="${EVAL_SAVE_LINK_PATH:-${SAVE_LINK_PATH}}"
EVAL_SAVE_NODE_PATH="${EVAL_SAVE_NODE_PATH:-${SAVE_NODE_PATH}}"
EVAL_SAVE_GRAPH_PATH="${EVAL_SAVE_GRAPH_PATH:-${SAVE_GRAPH_PATH}}"
EVAL_LOAD_LINK_MODEL="${EVAL_LOAD_LINK_MODEL:-${LOAD_LINK_MODEL}}"
EVAL_LOAD_NODE_MODEL="${EVAL_LOAD_NODE_MODEL:-${LOAD_NODE_MODEL}}"
EVAL_LOAD_GRAPH_MODEL="${EVAL_LOAD_GRAPH_MODEL:-${LOAD_GRAPH_MODEL}}"
EVAL_DATASET="${EVAL_DATASET:-${DATASET}}"
EVAL_FIXED_SPLIT="${EVAL_FIXED_SPLIT:-${FIXED_SPLIT}}"
EVAL_LINK_DATASET_SETTING="${EVAL_LINK_DATASET_SETTING:-${LINK_DATASET_SETTING}}"
EVAL_NODE_DATASET_SETTING="${EVAL_NODE_DATASET_SETTING:-${NODE_DATASET_SETTING}}"
EVAL_GRAPH_DATASET_SETTING="${EVAL_GRAPH_DATASET_SETTING:-${GRAPH_DATASET_SETTING}}"
EVAL_MIXED_DATASET_SETTING="${EVAL_MIXED_DATASET_SETTING:-${MIXED_DATASET_SETTING}}"
LINK_CSV="${LINK_CSV:-${OUTPUTS_DIR}/anygraph_link_eval.csv}"
NODE_CSV="${NODE_CSV:-${OUTPUTS_DIR}/anygraph_node_eval.csv}"
GRAPH_CSV="${GRAPH_CSV:-${OUTPUTS_DIR}/anygraph_graph_eval.csv}"
REPORT_CSV="${REPORT_CSV:-${OUTPUTS_DIR}/anygraph_report.csv}"
LINK_EDGE_EVAL_THRESHOLD_MODE="${LINK_EDGE_EVAL_THRESHOLD_MODE:-val_best_acc}"  # val_best_acc | zero
LINK_EDGE_EVAL_REPEAT_TIMES="${LINK_EDGE_EVAL_REPEAT_TIMES:-5}"

CMD=(
  python scripts/run_moe.py
  moe.method anygraph
  moe.anygraph.execution.step "${STEP}"
  moe.anygraph.paths.out_root "${OUT_ROOT}"
  device "${DEVICE}"
)

append_conversion_args() {
  CMD+=(
    moe.anygraph.conversion.task "${TASK}"
    moe.anygraph.paths.dataset_root "${DATASET_ROOT}"
    moe.anygraph.paths.split_root "${SPLIT_ROOT}"
    seeds "[${SEEDS}]"
    moe.anygraph.conversion.mask_col "${MASK_COL}"
    moe.anygraph.conversion.feat_dim "${FEAT_DIM}"
    moe.anygraph.conversion.edge_eval_payload_name "${EDGE_EVAL_PAYLOAD_NAME}"
    moe.anygraph.conversion.node_output_feat_dim "${NODE_OUTPUT_FEAT_DIM}"
  )

  if [ "${FEAT_REDUCTION}" = "1" ]; then
    CMD+=(moe.anygraph.conversion.feat_reduction True)
  else
    CMD+=(moe.anygraph.conversion.feat_reduction False)
  fi
  if [ "${L1_NORMALIZE_FEATURES}" = "1" ]; then
    CMD+=(moe.anygraph.conversion.l1_normalize_features True)
  else
    CMD+=(moe.anygraph.conversion.l1_normalize_features False)
  fi
  if [ "${EMIT_NODE_VAL}" = "1" ]; then
    CMD+=(moe.anygraph.conversion.emit_node_val True)
  else
    CMD+=(moe.anygraph.conversion.emit_node_val False)
  fi

  if [ -n "${DATASET}" ]; then
    CMD+=(moe.anygraph.conversion.dataset "${DATASET}")
  else
    CMD+=(moe.anygraph.conversion.dataset_file "${DATASET_FILE}")
  fi
  if [ -n "${SEEDS}" ]; then
    CMD+=(moe.anygraph.conversion.seeds "${SEEDS}")
  fi
  if [ -n "${EDGE_SPLITS}" ]; then
    CMD+=(moe.anygraph.conversion.edge_splits "${EDGE_SPLITS}")
  fi
  if [ -n "${NODE_SPLITS}" ]; then
    CMD+=(moe.anygraph.conversion.node_splits "${NODE_SPLITS}")
  fi
  if [ -n "${GRAPH_SPLITS}" ]; then
    CMD+=(moe.anygraph.conversion.graph_splits "${GRAPH_SPLITS}")
  fi
}

append_train_args() {
  CMD+=(
    moe.anygraph.train.mode "${TRAIN_MODE}"
    moe.anygraph.train.epoch "${TRAIN_EPOCH}"
    moe.anygraph.train.save_link_path "${TRAIN_SAVE_LINK_PATH}"
    moe.anygraph.train.save_node_path "${TRAIN_SAVE_NODE_PATH}"
    moe.anygraph.train.save_graph_path "${TRAIN_SAVE_GRAPH_PATH}"
  )

  if [ "${SKIP_IF_EXISTS}" = "1" ]; then
    CMD+=(moe.anygraph.train.skip_if_exists True)
  else
    CMD+=(moe.anygraph.train.skip_if_exists False)
  fi
  if [ -n "${TRAIN_LOAD_LINK_MODEL}" ]; then
    CMD+=(moe.anygraph.train.load_link_model "${TRAIN_LOAD_LINK_MODEL}")
  fi
  if [ -n "${TRAIN_LOAD_NODE_MODEL}" ]; then
    CMD+=(moe.anygraph.train.load_node_model "${TRAIN_LOAD_NODE_MODEL}")
  fi
  if [ -n "${TRAIN_LOAD_GRAPH_MODEL}" ]; then
    CMD+=(moe.anygraph.train.load_graph_model "${TRAIN_LOAD_GRAPH_MODEL}")
  fi
  if [ -n "${TRAIN_DATASET}" ]; then
    CMD+=(moe.anygraph.train.dataset.name "${TRAIN_DATASET}")
  fi
  if [ -n "${TRAIN_FIXED_SPLIT}" ]; then
    CMD+=(moe.anygraph.train.dataset.fixed_split "${TRAIN_FIXED_SPLIT}")
  fi
  if [ -n "${TRAIN_LINK_DATASET_SETTING}" ]; then
    CMD+=(moe.anygraph.train.link_dataset_setting "${TRAIN_LINK_DATASET_SETTING}")
  fi
  if [ -n "${TRAIN_NODE_DATASET_SETTING}" ]; then
    CMD+=(moe.anygraph.train.node_dataset_setting "${TRAIN_NODE_DATASET_SETTING}")
  fi
  if [ -n "${TRAIN_GRAPH_DATASET_SETTING}" ]; then
    CMD+=(moe.anygraph.train.graph_dataset_setting "${TRAIN_GRAPH_DATASET_SETTING}")
  fi
  if [ -n "${TRAIN_MIXED_DATASET_SETTING}" ]; then
    CMD+=(moe.anygraph.train.mixed_dataset_setting "${TRAIN_MIXED_DATASET_SETTING}")
  fi
}

append_eval_args() {
  CMD+=(
    moe.anygraph.train.save_link_path "${EVAL_SAVE_LINK_PATH}"
    moe.anygraph.train.save_node_path "${EVAL_SAVE_NODE_PATH}"
    moe.anygraph.train.save_graph_path "${EVAL_SAVE_GRAPH_PATH}"
    moe.anygraph.eval.mode "${EVAL_MODE}"
    moe.anygraph.output.link_csv "${LINK_CSV}"
    moe.anygraph.output.node_csv "${NODE_CSV}"
    moe.anygraph.output.graph_csv "${GRAPH_CSV}"
    moe.anygraph.output.report_csv "${REPORT_CSV}"
    moe.anygraph.conversion.edge_eval_payload_name "${EDGE_EVAL_PAYLOAD_NAME}"
    moe.anygraph.prediction.link.edge_eval_threshold_mode "${LINK_EDGE_EVAL_THRESHOLD_MODE}"
    moe.anygraph.prediction.link.edge_eval_repeat_times "${LINK_EDGE_EVAL_REPEAT_TIMES}"
  )

  if [ -n "${EVAL_LOAD_LINK_MODEL}" ]; then
    CMD+=(moe.anygraph.eval.load_link_model "${EVAL_LOAD_LINK_MODEL}")
  fi
  if [ -n "${EVAL_LOAD_NODE_MODEL}" ]; then
    CMD+=(moe.anygraph.eval.load_node_model "${EVAL_LOAD_NODE_MODEL}")
  fi
  if [ -n "${EVAL_LOAD_GRAPH_MODEL}" ]; then
    CMD+=(moe.anygraph.eval.load_graph_model "${EVAL_LOAD_GRAPH_MODEL}")
  fi
  if [ -n "${EVAL_DATASET}" ]; then
    CMD+=(moe.anygraph.eval.dataset.name "${EVAL_DATASET}")
  fi
  if [ -n "${EVAL_FIXED_SPLIT}" ]; then
    CMD+=(moe.anygraph.eval.dataset.fixed_split "${EVAL_FIXED_SPLIT}")
  fi
  if [ -n "${EVAL_LINK_DATASET_SETTING}" ]; then
    CMD+=(moe.anygraph.eval.link_dataset_setting "${EVAL_LINK_DATASET_SETTING}")
  fi
  if [ -n "${EVAL_NODE_DATASET_SETTING}" ]; then
    CMD+=(moe.anygraph.eval.node_dataset_setting "${EVAL_NODE_DATASET_SETTING}")
  fi
  if [ -n "${EVAL_GRAPH_DATASET_SETTING}" ]; then
    CMD+=(moe.anygraph.eval.graph_dataset_setting "${EVAL_GRAPH_DATASET_SETTING}")
  fi
  if [ -n "${EVAL_MIXED_DATASET_SETTING}" ]; then
    CMD+=(moe.anygraph.eval.mixed_dataset_setting "${EVAL_MIXED_DATASET_SETTING}")
  fi
}

case "${STEP}" in
  conversion)
    append_conversion_args
    ;;
  train)
    append_train_args
    ;;
  eval)
    append_eval_args
    ;;
  all)
    append_conversion_args
    append_train_args
    append_eval_args
    ;;
esac

if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "[AnyGraph] ROOT_DIR=${ROOT_DIR}"
echo "[AnyGraph] STEP=${STEP} OUT_ROOT=${OUT_ROOT} DEVICE=${DEVICE}"
echo "[AnyGraph] TRAIN_MODE=${TRAIN_MODE} EVAL_MODE=${EVAL_MODE} TRAIN_EPOCH=${TRAIN_EPOCH}"
echo "[AnyGraph] CMD=${CMD[*]}"
"${CMD[@]}"
