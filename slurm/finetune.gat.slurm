#!/bin/bash -l
#SBATCH --job-name "ft-gat"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --nodes=1               # one node per batch of finetune rows
#SBATCH --partition gpu         # partition
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --gpus-per-task=1       # one GPU per finetune task
#SBATCH --ntasks-per-node=4     # one task per GPU
#SBATCH --cpus-per-task=8       # cores per task (adjust if needed)
#SBATCH --array=0-269           # ceil(1080 rows / (4 tasks * 1 row/task)) = 270 array elements
#SBATCH --qos default           # SLURM qos
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

# TSV-driven gat model finetune experiments.
# Each SLURM task processes ROWS_PER_GPU rows sequentially. For each row it
# writes a temp single-row TSV and delegates to Python via finetune.run_tasks_tsv.
# The Python side uses an explicit pretrained_run_name when provided; otherwise
# it selects the matching pretrained checkpoint for cfg.seeds[0]. Each TSV row
# then expands to finetune.num_runs seeds. Account for this when sizing --time,
# ROWS_PER_GPU, and --array.
#
# Row format (slurm/finetune.gat.all.tsv):
#   Header: model pretrain_dataset pretrain_task_level pretrain_induced pretrain_method dataset task_level induced task_type fixed_split finetune_method graphprompt_plus gpf_plus edgeprompt_plus
#   Rows may omit trailing columns; missing values use config defaults.
# Row count changes over time; the srun script errors out when --array is undersized.

source "${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"

EXPERIMENT_FILE="${ROOT_DIR}/slurm/finetune.gat.all.tsv"
if [ ! -f "${EXPERIMENT_FILE}" ]; then
  echo "[finetune] ERROR: Experiment file not found: ${EXPERIMENT_FILE}" >&2
  exit 2
fi
EXPERIMENT_COLUMNS="dataset|task_level|induced|task_type|fixed_split|pretrained_run_name|finetune_method|model|pretrain_dataset|pretrain_task_level|pretrain_induced|pretrain_method|graphprompt_plus|gpf_plus|edgeprompt_plus|skip_if_exists"
parse_experiment_file "${EXPERIMENT_FILE}" "${EXPERIMENT_COLUMNS}" "[finetune]"

export EXPERIMENT_FILE EXPERIMENT_COLUMNS

TASK_COUNT="${SLURM_NTASKS:-${SLURM_NTASKS_PER_NODE:-4}}"
ROWS_PER_GPU="${ROWS_PER_GPU:-1}"
export ROWS_PER_GPU
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:---label}"
FAILURE_DIR="$(mktemp -d "${ROOT_DIR}/slurm/output/${SLURM_JOB_NAME:-finetune}.${SLURM_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID:-0}.failures.XXXXXX")"
trap 'rm -rf "${FAILURE_DIR:?}"' EXIT
export FAILURE_DIR

if srun ${SRUN_EXTRA_ARGS} --ntasks="${TASK_COUNT}" --kill-on-bad-exit=0 bash <<'SRUN_SCRIPT'
set -euo pipefail

ROOT_DIR="${ROOT_DIR:?}"
source "${ROOT_DIR}/slurm/_common.sh"
EXPERIMENT_FILE="${EXPERIMENT_FILE:?}"
EXPERIMENT_COLUMNS="${EXPERIMENT_COLUMNS:?}"
FAILURE_DIR="${FAILURE_DIR:?}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_ID="${SLURM_PROCID:-0}"
TOTAL_TASKS="${SLURM_NTASKS:-${SLURM_NTASKS_PER_NODE:-4}}"
ROWS_PER_GPU="${ROWS_PER_GPU:-1}"
if ! [[ "${ROWS_PER_GPU}" =~ ^[0-9]+$ ]] || (( ROWS_PER_GPU < 1 )); then
  echo "[finetune] ERROR: ROWS_PER_GPU must be a positive integer (got '${ROWS_PER_GPU}')" >&2
  exit 2
fi
BATCH_SIZE=$(( TOTAL_TASKS * ROWS_PER_GPU ))
BASE_IDX=$(( ARRAY_ID * BATCH_SIZE ))
HOST="$(hostname)"

# Parse experiment file and detect header.
parse_experiment_file "${EXPERIMENT_FILE}" \
  "${EXPERIMENT_COLUMNS}" \
  "[finetune]"

log() {
  echo "[$(date +%F' '%T)] [array=${ARRAY_ID} task=${TASK_ID}/${TOTAL_TASKS} rows_per_gpu=${ROWS_PER_GPU} global=${GLOBAL_IDX:-na} host=${HOST} cuda=${CUDA_VISIBLE_DEVICES:-unset}] $*"
}

# Validate array range.
EXPECTED_ARRAYS=$(( (TOTAL_ROWS + BATCH_SIZE - 1) / BATCH_SIZE ))
if (( ARRAY_ID >= EXPECTED_ARRAYS )); then
  log "WARN: array_id=${ARRAY_ID} exceeds needed elements (rows=${TOTAL_ROWS}, batch=${BATCH_SIZE}, need=${EXPECTED_ARRAYS})"
fi
# The reverse mismatch silently drops trailing TSV rows: record a failure
# marker (matched by the row_*.failure glob) so the job exits non-zero.
ARRAY_MAX="${SLURM_ARRAY_TASK_MAX:-${ARRAY_ID}}"
if (( ARRAY_MAX + 1 < EXPECTED_ARRAYS )) && (( ARRAY_ID == 0 )) && (( TASK_ID == 0 )); then
  log "ERROR: --array covers $(( ARRAY_MAX + 1 )) element(s) but ${TOTAL_ROWS} rows need ${EXPECTED_ARRAYS}; resize to --array=0-$(( EXPECTED_ARRAYS - 1 ))"
  printf 'undersized --array: have=%s need=%s rows=%s\n' \
    "$(( ARRAY_MAX + 1 ))" "${EXPECTED_ARRAYS}" "${TOTAL_ROWS}" \
    > "${FAILURE_DIR}/row_undersized_array.failure"
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
  # Python-side run_tasks_tsv path handles parsing, checkpoint discovery,
  # and multi-seed expansion.
  TMPTSV="$(mktemp /tmp/finetune_row_XXXXXX.tsv)"
  { [[ -n "${HEADER_LINE}" ]] && echo "${HEADER_LINE}"; echo "${ROW}"; } > "${TMPTSV}"

  log "Starting row ${GLOBAL_IDX}/${TOTAL_ROWS}: ${ROW}"
  # A failure in one row must not abort remaining rows in the same task.
  set +e
  python "${ROOT_DIR}/scripts/run_finetune.py" \
    finetune.run_tasks_tsv True \
    finetune.tasks_tsv "${TMPTSV}" \
    finetune.num_runs 5 \
    device 0
  RC=$?
  set -e
  rm -f "${TMPTSV}"
  if (( RC == 0 )); then
    log "Finished row ${GLOBAL_IDX} successfully"
  else
    FAILED_ROWS=$(( FAILED_ROWS + 1 ))
    printf 'row=%s rc=%s task=%s host=%s cuda=%s\n%s\n' \
      "${GLOBAL_IDX}" "${RC}" "${TASK_ID}" "${HOST}" "${CUDA_VISIBLE_DEVICES:-unset}" "${ROW}" \
      > "${FAILURE_DIR}/row_${GLOBAL_IDX}_task_${TASK_ID}.failure"
    log "FAILED row ${GLOBAL_IDX} with rc=${RC}"
  fi
done

if (( ASSIGNED_ROWS == 0 )); then
  log "No row assigned; total_rows=${TOTAL_ROWS}"
else
  log "Finished ${ASSIGNED_ROWS} assigned row(s); failed=${FAILED_ROWS}"
fi

if (( FAILED_ROWS > 0 )); then
  log "Recorded failed row marker(s) in ${FAILURE_DIR}; allowing sibling tasks to finish"
fi
SRUN_SCRIPT
then
  if compgen -G "${FAILURE_DIR}/row_*.failure" > /dev/null; then
    echo "[finetune] ERROR: one or more rows failed for array_id=${SLURM_ARRAY_TASK_ID:-0}; sibling tasks were allowed to finish first" >&2
    sed 's/^/[finetune]   /' "${FAILURE_DIR}"/row_*.failure >&2
    exit 1
  fi
else
  rc=$?
  echo "[finetune] ERROR: srun failed with rc=${rc} for array_id=${SLURM_ARRAY_TASK_ID:-0}" >&2
  exit "${rc}"
fi
