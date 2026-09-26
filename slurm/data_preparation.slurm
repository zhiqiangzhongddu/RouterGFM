#!/bin/bash -l
#SBATCH --job-name "dp"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --partition gpu         # partition
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # one dataset per array task
#SBATCH --gpus=1                # one GPU per dataset
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-62            # 33 node + 30 graph = 63 datasets (indices 0-62)
#SBATCH --qos default
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"

NODE_DATASETS_TSV="${NODE_DATASETS_TSV:-${ROOT_DIR}/data/available_node_datasets.tsv}"
GRAPH_DATASETS_TSV="${GRAPH_DATASETS_TSV:-${ROOT_DIR}/data/available_graph_datasets.tsv}"

if [ ! -f "${NODE_DATASETS_TSV}" ]; then
  echo "[data-preparation] ERROR: Missing node dataset list: ${NODE_DATASETS_TSV}"
  exit 1
fi
if [ ! -f "${GRAPH_DATASETS_TSV}" ]; then
  echo "[data-preparation] ERROR: Missing graph dataset list: ${GRAPH_DATASETS_TSV}"
  exit 1
fi

TMP_DATASET_ROWS="$(mktemp)"
TMP_SINGLE_DATASET="$(mktemp)"
trap 'rm -f "${TMP_DATASET_ROWS}" "${TMP_SINGLE_DATASET}"' EXIT

# Build one merged list so each array task handles one dataset.
{
  grep -v '^[#[:space:]]' "${NODE_DATASETS_TSV}" | sed 's/$/\tnode-edge/'
  grep -v '^[#[:space:]]' "${GRAPH_DATASETS_TSV}" | sed 's/$/\tgraph/'
} > "${TMP_DATASET_ROWS}"

mapfile -t DATASET_ROWS < "${TMP_DATASET_ROWS}"
TOTAL=${#DATASET_ROWS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK_ID >= TOTAL )); then
  echo "[data-preparation] WARN: SLURM_ARRAY_TASK_ID=${TASK_ID} exceeds dataset count (${TOTAL}); skipping."
  exit 0
fi

IFS=$'\t' read -r DATASET DATASET_KIND <<< "${DATASET_ROWS[TASK_ID]}"
printf '%s\n' "${DATASET}" > "${TMP_SINGLE_DATASET}"

echo "========================================================================"
echo "[SLURM][DataPrep] task=${TASK_ID}/${TOTAL} dataset=${DATASET} kind=${DATASET_KIND}"
echo "========================================================================"

CMD=(
  python "${ROOT_DIR}/scripts/run_data_preparation.py"
  data_preparation.target_datasets "${TMP_SINGLE_DATASET}"
  data_preparation.dataset.root "${ROOT_DIR}/data/datasets"
)
TASK_LEVEL_OVERRIDE=""
NODE_ONLY_DATASETS="${NODE_ONLY_DATASETS:-}"
NODE_ONLY_DATASETS_NORMALIZED=",${NODE_ONLY_DATASETS// /},"
if [ -n "${NODE_ONLY_DATASETS}" ] \
  && [[ "${NODE_ONLY_DATASETS_NORMALIZED}" == *",${DATASET},"* ]]; then
  TASK_LEVEL_OVERRIDE="node"
fi
EDGE_ONLY_DATASETS="${EDGE_ONLY_DATASETS:-}"
EDGE_ONLY_DATASETS_NORMALIZED=",${EDGE_ONLY_DATASETS// /},"
if [ -n "${EDGE_ONLY_DATASETS}" ] \
  && [[ "${EDGE_ONLY_DATASETS_NORMALIZED}" == *",${DATASET},"* ]]; then
  if [ -n "${TASK_LEVEL_OVERRIDE}" ]; then
    echo "[data-preparation] ERROR: ${DATASET} is listed in both NODE_ONLY_DATASETS and EDGE_ONLY_DATASETS." >&2
    exit 2
  fi
  TASK_LEVEL_OVERRIDE="edge"
fi
if [ -n "${TASK_LEVEL_OVERRIDE}" ]; then
  CMD+=(data_preparation.task_level_override "${TASK_LEVEL_OVERRIDE}")
fi
FORCE_EDGE_DATASETS="${FORCE_EDGE_DATASETS:-}"
FORCE_EDGE_DATASETS_NORMALIZED=",${FORCE_EDGE_DATASETS// /},"
if [ -n "${FORCE_EDGE_DATASETS}" ] \
  && [[ "${FORCE_EDGE_DATASETS_NORMALIZED}" == *",${DATASET},"* ]]; then
  CMD+=(
    data_preparation.dataset.edge_level_max_num_nodes
    "${EDGE_LEVEL_MAX_NUM_NODES_OVR:-0}"
  )
fi
srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  "${CMD[@]}"
