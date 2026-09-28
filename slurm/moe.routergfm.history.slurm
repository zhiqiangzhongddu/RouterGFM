#!/bin/bash -l
#SBATCH --job-name "routergfm_history"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --partition gpu         # partition
#SBATCH --nodes=1               # one expert shard per array element
#SBATCH --ntasks=1
#SBATCH --gpus=1                # frozen encoders and heads run on one GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-47            # moe.routergfm.experts.num_shards = array size (588 experts -> ~12 per shard)
#SBATCH --qos default           # SLURM qos
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

# RouterGFM historical evaluations (paper Alg. 1 l.1-4).
# Array element i records the experts catalog[i::NUM_SHARDS] on every declared
# application (moe.routergfm.experts.shard_index = i). Existing records are
# skipped, so failed shards can simply be resubmitted. Shard 0 also writes the
# label-free descriptors of every data key.
#
# NUM_SHARDS defaults to the submitted array size (SLURM_ARRAY_TASK_MAX + 1).
# When resubmitting a subset (e.g. --array=5,17), export NUM_SHARDS=48 so the
# expert partition is unchanged. Extra config overrides are passed through
# EXTRA_OVERRIDES="key value ...".

source "${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-$(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 ))}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
if ! [[ "${NUM_SHARDS}" =~ ^[0-9]+$ ]] || (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "[routergfm-history] ERROR: shard ${SHARD_INDEX} is outside NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

echo "[$(date +%F' '%T)] [routergfm-history] shard ${SHARD_INDEX}/${NUM_SHARDS} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
python "${ROOT_DIR}/scripts/run_moe.py" \
  moe.method routergfm \
  moe.routergfm.task history \
  moe.routergfm.experts.shard_index "${SHARD_INDEX}" \
  moe.routergfm.experts.num_shards "${NUM_SHARDS}" \
  device 0 \
  ${EXTRA_OVERRIDES}
