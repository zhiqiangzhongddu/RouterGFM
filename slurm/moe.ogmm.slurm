#!/bin/bash -l
#SBATCH --job-name "moe_ogmm"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --nodes=1               # one node per batch of ogmm rows
#SBATCH --partition gpu         # partition
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --gpus-per-task=1       # one GPU per task
#SBATCH --ntasks-per-node=4     # one task per GPU
#SBATCH --cpus-per-task=8       # cores per task (adjust if needed)
#SBATCH --array=0-13            # ceil(54 rows / (4 tasks * 1 row/task)) = 14 array elements
#SBATCH --qos default           # SLURM qos
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

# TSV-driven OGMM experiments (RouterGFM Table 15 shift comparison).
# Each SLURM task processes ROWS_PER_GPU rows sequentially. For each row it
# writes a temp single-row TSV and delegates to Python via
# scripts/run_moe.py moe.method ogmm moe.ogmm.run_tasks_tsv True.
#
# Shift rows (split_root under data/splits_shift/<condition>) are verified
# first with `python -m src.data_loader.shift_splits ... shift.verify True`:
# a missing or regenerated shift file would otherwise be silently replaced by
# a standard random split. Build the shift roots before submitting.
#
# Row format (slurm/moe.ogmm.all.tsv):
#   Header: dataset task_level induced (+ optional: task_type fixed_split
#           split_root num_domains top_k expert_epochs gen_epochs
#           merge_epochs batch skip_if_exists)
#   Rows may omit trailing columns; missing values use config defaults.

source "${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"

EXPERIMENT_FILE="${ROOT_DIR}/slurm/moe.ogmm.all.tsv"
if [ ! -f "${EXPERIMENT_FILE}" ]; then
  echo "[ogmm] ERROR: Experiment file not found: ${EXPERIMENT_FILE}" >&2
  exit 2
fi

EXPERIMENT_COLUMNS="dataset|task_level|task_type|induced|fixed_split|split_root|num_domains|top_k|expert_epochs|gen_epochs|merge_epochs|batch|skip_if_exists"
parse_experiment_file "${EXPERIMENT_FILE}" "${EXPERIMENT_COLUMNS}" "[ogmm]"

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
  echo "[ogmm] ERROR: ROWS_PER_GPU must be a positive integer (got '${ROWS_PER_GPU}')" >&2
  exit 2
fi
BATCH_SIZE=$(( TOTAL_TASKS * ROWS_PER_GPU ))
BASE_IDX=$(( ARRAY_ID * BATCH_SIZE ))
HOST="$(hostname)"

# Parse experiment file and detect header.
parse_experiment_file "${EXPERIMENT_FILE}" \
  "${EXPERIMENT_COLUMNS}" \
  "[ogmm]"

log() {
  echo "[$(date +%F' '%T)] [array=${ARRAY_ID} task=${TASK_ID}/${TOTAL_TASKS} rows_per_gpu=${ROWS_PER_GPU} global=${GLOBAL_IDX:-na} host=${HOST} cuda=${CUDA_VISIBLE_DEVICES:-unset}] $*"
}

# Value of header column $2 in tab-separated row $1 (empty when absent).
row_value() {
  local -a _cols _vals
  local _i
  read -r -a _cols <<< "${HEADER_LINE}"
  IFS=$'\t' read -r -a _vals <<< "$1"
  for _i in "${!_cols[@]}"; do
    if [[ "${_cols[_i],,}" == "$2" ]]; then
      printf '%s' "${_vals[_i]:-}"
      return 0
    fi
  done
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

  SPLIT_ROOT="$(row_value "${ROW}" split_root)"
  if [[ "${SPLIT_ROOT}" == *splits_shift/* ]]; then
    set +e
    python -m src.data_loader.shift_splits \
      data_preparation.target_datasets "$(row_value "${ROW}" dataset)" \
      data_preparation.shift.root "$(dirname "${SPLIT_ROOT}")" \
      data_preparation.shift.conditions "['$(basename "${SPLIT_ROOT}")']" \
      data_preparation.shift.verify True
    RC=$?
    set -e
    if (( RC != 0 )); then
      FAILED_ROWS=$(( FAILED_ROWS + 1 ))
      log "FAILED row ${GLOBAL_IDX}: shift splits under ${SPLIT_ROOT} did not verify; not running it"
      continue
    fi
  fi

  # Write header (if present) + assigned row into a temp TSV in /tmp so the
  # Python-side run_tasks_tsv path handles parsing, validation, and result saving.
  TMPTSV="$(mktemp /tmp/ogmm_row_XXXXXX.tsv)"
  { [[ -n "${HEADER_LINE}" ]] && echo "${HEADER_LINE}"; echo "${ROW}"; } > "${TMPTSV}"

  log "Starting row ${GLOBAL_IDX}/${TOTAL_ROWS}: ${ROW}"
  # A failure in one row must not abort remaining rows in the same task.
  set +e
  python "${ROOT_DIR}/scripts/run_moe.py" \
    moe.method ogmm \
    moe.ogmm.run_tasks_tsv True \
    moe.ogmm.tasks_tsv "${TMPTSV}" \
    moe.ogmm.num_runs "${NUM_RUNS}" \
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
  echo "[ogmm] ERROR: srun failed with rc=${rc} for array_id=${SLURM_ARRAY_TASK_ID:-0}" >&2
  exit "${rc}"
fi
