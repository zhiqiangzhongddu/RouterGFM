#!/bin/bash -l
#SBATCH --job-name "routergfm_baselines"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --nodes=1               # one node per batch of baseline rows
#SBATCH --partition gpu         # partition
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --gpus-per-task=1       # one GPU per task
#SBATCH --ntasks-per-node=4     # one task per GPU
#SBATCH --cpus-per-task=8       # cores per task (adjust if needed)
#SBATCH --array=0-15            # ceil(62 rows / (4 tasks * 1 row/task)) = 16 array elements
#SBATCH --qos default           # SLURM qos
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

# TSV-driven RouterGFM matched-pool baselines (App. C: MetaGL-U, SAGMM-PE,
# META-DES, KDEM, PPEM). Each row runs the row's method on (dataset, budget) for
# seeds moe.routergfm.apps.seeds[:NUM_RUNS] over the shared RouterInfra
# (history, heads, embeddings under moe.routergfm.output_root must exist) and
# appends one row to outputs/results/moe_<method>.tsv. Each SLURM task handles
# ROWS_PER_GPU rows through a temp single-row TSV
# (moe.routergfm.task matched_baseline, moe.routergfm.baselines.run_tasks_tsv True).
#
# Row format (slurm/moe.routergfm_baselines.tsv):
#   Header: method dataset task_level budget
# Extra config overrides: EXTRA_OVERRIDES="moe.routergfm.baselines.candidate_rule eligible".

source "${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"

EXPERIMENT_FILE="${ROOT_DIR}/slurm/moe.routergfm_baselines.tsv"
if [ ! -f "${EXPERIMENT_FILE}" ]; then
  echo "[routergfm-baselines] ERROR: Experiment file not found: ${EXPERIMENT_FILE}" >&2
  exit 2
fi

EXPERIMENT_COLUMNS="method|dataset|task_level|budget"
parse_experiment_file "${EXPERIMENT_FILE}" "${EXPERIMENT_COLUMNS}" "[routergfm-baselines]"

export EXPERIMENT_FILE EXPERIMENT_COLUMNS

TASK_COUNT="${SLURM_NTASKS:-${SLURM_NTASKS_PER_NODE:-4}}"
ROWS_PER_GPU="${ROWS_PER_GPU:-1}"
export ROWS_PER_GPU
NUM_RUNS="${NUM_RUNS:-5}"
export NUM_RUNS
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"  # extra "key value" config overrides (word-split)
export EXTRA_OVERRIDES
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:---label}"

if srun ${SRUN_EXTRA_ARGS} --ntasks="${TASK_COUNT}" --kill-on-bad-exit=0 bash <<'SRUN_SCRIPT'
set -euo pipefail

ROOT_DIR="${ROOT_DIR:?}"
source "${ROOT_DIR}/slurm/_common.sh"
EXPERIMENT_FILE="${EXPERIMENT_FILE:?}"
EXPERIMENT_COLUMNS="${EXPERIMENT_COLUMNS:?}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_ID="${SLURM_PROCID:-0}"
TOTAL_TASKS="${SLURM_NTASKS:-${SLURM_NTASKS_PER_NODE:-4}}"
ROWS_PER_GPU="${ROWS_PER_GPU:-1}"
NUM_RUNS="${NUM_RUNS:-5}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
if ! [[ "${ROWS_PER_GPU}" =~ ^[0-9]+$ ]] || (( ROWS_PER_GPU < 1 )); then
  echo "[routergfm-baselines] ERROR: ROWS_PER_GPU must be a positive integer (got '${ROWS_PER_GPU}')" >&2
  exit 2
fi
BATCH_SIZE=$(( TOTAL_TASKS * ROWS_PER_GPU ))
BASE_IDX=$(( ARRAY_ID * BATCH_SIZE ))
HOST="$(hostname)"

# Parse experiment file and detect header.
parse_experiment_file "${EXPERIMENT_FILE}" \
  "${EXPERIMENT_COLUMNS}" \
  "[routergfm-baselines]"
read -r -a HEADER_COLS <<< "${HEADER_LINE}"

log() {
  echo "[$(date +%F' '%T)] [array=${ARRAY_ID} task=${TASK_ID}/${TOTAL_TASKS} rows_per_gpu=${ROWS_PER_GPU} global=${GLOBAL_IDX:-na} host=${HOST} cuda=${CUDA_VISIBLE_DEVICES:-unset}] $*"
}

# Validate array range.
EXPECTED_ARRAYS=$(( (TOTAL_ROWS + BATCH_SIZE - 1) / BATCH_SIZE ))
if (( ARRAY_ID >= EXPECTED_ARRAYS )); then
  log "WARN: array_id=${ARRAY_ID} exceeds needed elements (rows=${TOTAL_ROWS}, batch=${BATCH_SIZE}, need=${EXPECTED_ARRAYS})"
fi
# The reverse mismatch silently drops trailing TSV rows — make it unmissable.
ARRAY_MAX="${SLURM_ARRAY_TASK_MAX:-${ARRAY_ID}}"
if (( ARRAY_MAX + 1 < EXPECTED_ARRAYS )); then
  log "ERROR: --array covers $(( ARRAY_MAX + 1 )) element(s) but ${TOTAL_ROWS} rows need ${EXPECTED_ARRAYS}; trailing rows will NOT run. Resize to --array=0-$(( EXPECTED_ARRAYS - 1 ))."
  exit 2
fi

cd "${ROOT_DIR}"
log "Using python=$(command -v python) conda_prefix=${CONDA_PREFIX:-unset}"

ASSIGNED_ROWS=0
FAILED_ROWS=0
for ((ROW_SLOT=0; ROW_SLOT<ROWS_PER_GPU; ROW_SLOT++)); do
  GLOBAL_IDX=$(( BASE_IDX + ROW_SLOT * TOTAL_TASKS + TASK_ID ))
  if (( GLOBAL_IDX >= TOTAL_ROWS )); then
    continue
  fi

  ASSIGNED_ROWS=$(( ASSIGNED_ROWS + 1 ))
  ROW="${ROWS[GLOBAL_IDX]}"

  # The row's method column selects the baseline (moe.routergfm.baselines.method).
  read -r -a ROW_COLS <<< "${ROW}"
  METHOD=""
  for COL_IDX in "${!HEADER_COLS[@]}"; do
    if [[ "${HEADER_COLS[COL_IDX],,}" == "method" ]]; then
      METHOD="${ROW_COLS[COL_IDX]:-}"
    fi
  done
  if [[ -z "${METHOD}" ]]; then
    FAILED_ROWS=$(( FAILED_ROWS + 1 ))
    log "FAILED row ${GLOBAL_IDX}: no method value (header: method dataset task_level budget) in '${ROW}'"
    continue
  fi

  # Write header (if present) + assigned row into a temp TSV in /tmp so the
  # Python-side run_tasks_tsv path handles parsing, validation, and result saving.
  TMPTSV="$(mktemp /tmp/routergfm_baselines_row_XXXXXX.tsv)"
  { [[ -n "${HEADER_LINE}" ]] && echo "${HEADER_LINE}"; echo "${ROW}"; } > "${TMPTSV}"

  log "Starting row ${GLOBAL_IDX}/${TOTAL_ROWS}: ${ROW}"
  # A failure in one row must not abort remaining rows in the same task.
  set +e
  python "${ROOT_DIR}/scripts/run_moe.py" \
    moe.method routergfm \
    moe.routergfm.task matched_baseline \
    moe.routergfm.baselines.method "${METHOD}" \
    moe.routergfm.baselines.run_tasks_tsv True \
    moe.routergfm.baselines.tasks_tsv "${TMPTSV}" \
    moe.routergfm.baselines.num_runs "${NUM_RUNS}" \
    device 0 \
    ${EXTRA_OVERRIDES}
  RC=$?
  set -e
  rm -f "${TMPTSV}"
  if (( RC == 0 )); then
    log "Finished row ${GLOBAL_IDX} successfully"
  else
    FAILED_ROWS=$(( FAILED_ROWS + 1 ))
    log "FAILED row ${GLOBAL_IDX} with rc=${RC}"
  fi
done

if (( ASSIGNED_ROWS == 0 )); then
  log "No row assigned; total_rows=${TOTAL_ROWS}"
else
  log "Finished ${ASSIGNED_ROWS} assigned row(s); failed=${FAILED_ROWS}"
fi

if (( FAILED_ROWS > 0 )); then
  exit 1
fi
SRUN_SCRIPT
then
  :
else
  rc=$?
  echo "[routergfm-baselines] ERROR: srun failed with rc=${rc} for array_id=${SLURM_ARRAY_TASK_ID:-0}" >&2
  exit "${rc}"
fi
