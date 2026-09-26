#!/bin/bash
# Shared SLURM preamble for all workflow scripts.
# Source this file from each *.slurm script after set -euo pipefail:
#
#   set -euo pipefail
#   source "${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"
#
# Provides:
#   ROOT_DIR        — resolved project root (exported)
#   PYTHONPATH      — includes ROOT_DIR (exported)
#   Conda env       — activated (CONDA_ENV, default "agae")
#   slurm/output/   — created

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
if [ "$(basename "${ROOT_DIR}")" = "slurm" ]; then
  ROOT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
fi
export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
# The former default (/project/home/p201211/conda_base_path) does not exist,
# so every batch job died at "conda: command not found". The real root sits
# one level deeper; export CONDA_BASE to override on another machine.
CONDA_BASE="${CONDA_BASE:-/project/home/p201211/zzhong/conda_base_path/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV:-agae}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

# ---------------------------------------------------------------------------
# Ensure output directory exists
# ---------------------------------------------------------------------------
mkdir -p "${ROOT_DIR}/slurm/output"
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# parse_experiment_file FILE "col1|col2|..." PREFIX
#
# Load the TSV experiment file, detect an optional header (commented or
# uncommented), and populate the following variables in the caller's scope:
#   ALL_LINES  — array of all non-empty, non-comment data lines
#   HEADER_LINE — detected header (empty string if none)
#   ROWS       — array of data rows (ALL_LINES minus header, if detected)
#   TOTAL_ROWS — number of data rows
# If no data rows remain, the calling launcher exits successfully before
# starting workers. A zero-row SLURM array is invalid, so batch launchers keep
# a minimal 0-0 array and turn the single task into this clean no-op instead.
#
# Column names are supplied as a pipe-separated string that maps directly
# to a bash case pattern, e.g. "model|dataset|task_level|induced|method".
# ---------------------------------------------------------------------------
parse_experiment_file() {
  local _file="$1"
  local _columns="$2"
  local _prefix="${3:-[parse]}"

  mapfile -t ALL_LINES < <(grep -Ev '^[[:space:]]*($|#)' "${_file}")

  # Look for a #-prefixed header line.
  HEADER_LINE=""
  while IFS= read -r _cline; do
    local _body="${_cline#\#}"
    _body="${_body## }"
    [[ -z "${_body}" ]] && continue
    local _is_header=true
    for _tok in ${_body}; do
      local _low="${_tok,,}"
      case "|${_columns}|" in
        *"|${_low}|"*) ;;
        *) _is_header=false; break ;;
      esac
    done
    if ${_is_header}; then
      HEADER_LINE="${_body}"
    fi
    break
  done < <(grep -E '^[[:space:]]*#' "${_file}" | head -1)

  # Also check if first data line is an uncommented header (backward compat).
  local _data_start=0
  if [[ -z "${HEADER_LINE}" ]] && (( ${#ALL_LINES[@]} > 0 )); then
    local _first="${ALL_LINES[0]}"
    local _is_header=true
    for _tok in ${_first}; do
      local _low="${_tok,,}"
      case "|${_columns}|" in
        *"|${_low}|"*) ;;
        *) _is_header=false; break ;;
      esac
    done
    if ${_is_header}; then
      HEADER_LINE="${_first}"
      _data_start=1
    fi
  fi

  ROWS=("${ALL_LINES[@]:${_data_start}}")
  TOTAL_ROWS="${#ROWS[@]}"

  if (( TOTAL_ROWS == 0 )); then
    echo "${_prefix} No data rows in experiment file (after stripping comments/header); nothing to run."
    exit 0
  fi
}
