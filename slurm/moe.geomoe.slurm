#!/bin/bash -l
#SBATCH --job-name "moe_geomoe"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --nodes=1               # one node per batch of geomoe rows
#SBATCH --partition gpu         # partition
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --gpus-per-task=1       # one GPU per task
#SBATCH --ntasks-per-node=4     # one task per GPU
#SBATCH --cpus-per-task=8       # cores per task (adjust if needed)
#SBATCH --array=0-9             # ceil(40 rows / (4 tasks * 1 row/task)) = 10 array elements
#SBATCH --qos default           # SLURM qos
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

# TSV-driven GeoMoE experiments.
# Each SLURM task processes ROWS_PER_GPU rows sequentially. For each row it
# writes a temp single-row TSV and delegates to Python via
# scripts/run_moe.py moe.method geomoe moe.geomoe.run_tasks_tsv True. This reuses
# the Python-side parser, validation, and result-saving logic while keeping the
# number of submitted array elements small.
#
# Row format (slurm/moe.geomoe.all.tsv):
#   Header: dataset task_level induced (+ optional: task_type fixed_split
#           split_root epochs batch skip_if_exists)
#   Rows may omit trailing columns; missing values use config defaults.
#   split_root selects the shift condition (data/splits_shift/<condition>).
#   Build the shift roots first (src/data_loader/shift_splits.py). Every run
#   verifies its shift split file before loading and fails if it is missing
#   or was regenerated as a standard split.

source "${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"

EXPERIMENT_FILE="${ROOT_DIR}/slurm/moe.geomoe.all.tsv"
if [ ! -f "${EXPERIMENT_FILE}" ]; then
  echo "[geomoe] ERROR: Experiment file not found: ${EXPERIMENT_FILE}" >&2
  exit 2
fi

EXPERIMENT_COLUMNS="dataset|task_level|task_type|induced|fixed_split|split_root|epochs|batch|skip_if_exists"
parse_experiment_file "${EXPERIMENT_FILE}" "${EXPERIMENT_COLUMNS}" "[geomoe]"

export EXPERIMENT_FILE EXPERIMENT_COLUMNS

TASK_COUNT="${SLURM_NTASKS:-${SLURM_NTASKS_PER_NODE:-4}}"
ROWS_PER_GPU="${ROWS_PER_GPU:-1}"
export ROWS_PER_GPU
NUM_RUNS="${NUM_RUNS:-5}"
export NUM_RUNS
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
if ! [[ "${ROWS_PER_GPU}" =~ ^[0-9]+$ ]] || (( ROWS_PER_GPU < 1 )); then
  echo "[geomoe] ERROR: ROWS_PER_GPU must be a positive integer (got '${ROWS_PER_GPU}')" >&2
  exit 2
fi
BATCH_SIZE=$(( TOTAL_TASKS * ROWS_PER_GPU ))
BASE_IDX=$(( ARRAY_ID * BATCH_SIZE ))
HOST="$(hostname)"

# Parse experiment file and detect header.
parse_experiment_file "${EXPERIMENT_FILE}" \
  "${EXPERIMENT_COLUMNS}" \
  "[geomoe]"

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

  # Write header (if present) + assigned row into a temp TSV in /tmp so the
  # Python-side run_tasks_tsv path handles parsing, validation, and result saving.
  TMPTSV="$(mktemp /tmp/geomoe_row_XXXXXX.tsv)"
  { [[ -n "${HEADER_LINE}" ]] && echo "${HEADER_LINE}"; echo "${ROW}"; } > "${TMPTSV}"

  log "Starting row ${GLOBAL_IDX}/${TOTAL_ROWS}: ${ROW}"
  # A failure in one row must not abort remaining rows in the same task.
  set +e
  python "${ROOT_DIR}/scripts/run_moe.py" \
    moe.method geomoe \
    moe.geomoe.run_tasks_tsv True \
    moe.geomoe.tasks_tsv "${TMPTSV}" \
    moe.geomoe.num_runs "${NUM_RUNS}" \
    device 0
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
  echo "[geomoe] ERROR: srun failed with rc=${rc} for array_id=${SLURM_ARRAY_TASK_ID:-0}" >&2
  exit "${rc}"
fi
