#!/usr/bin/env bash
# Evaluate a trained checkpoint on the configured eval set.
# Usage:
#   bash scripts/evaluate.sh <config_alias> <last_policy_path> [--benchmark aime24] [extra args...]
#
# Examples:
#   bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH "${INGPO_ROOT}/experiments/exp1-ingpo-tree-666-qwen1.5b-math/iteration_0010"
#   bash scripts/evaluate.sh polIter_rho1bSft2_ingpo_tree_GSM8K  "<hf-checkpoint>"
#   bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH "<hf-or-local-checkpoint>" --benchmark aime24

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
  cat <<EOF
Usage: bash scripts/evaluate.sh <config_alias> <last_policy_path> [--benchmark all|aime24] [extra args...]

Examples:
  bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH experiments/run/iteration_0010 --benchmark aime24
  INGPO_EVAL_BENCHMARK=aime24 bash scripts/evaluate.sh qwen1_5b_base_for_MATH_eval Qwen/Qwen2.5-1.5B

When --benchmark aime24 is used, this script keeps only the aime24_test
pipeline and ensures data/aime24 exists by running scripts/download_eval_datasets.py.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CFG_NAME="${1:?Usage: evaluate.sh <config_alias> <last_policy_path> [--benchmark all|aime24] [extra args...]}"
LAST_POLICY="${2:?missing last_policy_path}"
shift 2 || true

CFG="${INGPO_ROOT}/configs/${CFG_NAME}.jsonnet"
[[ -f "${CFG}" ]] || { echo "Cannot find config ${CFG_NAME}"; exit 2; }

BENCHMARK="${INGPO_EVAL_BENCHMARK:-all}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark|--dataset|--task)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 2; }
      BENCHMARK="$2"
      shift 2
      ;;
    --benchmark=*|--dataset=*|--task=*)
      BENCHMARK="${1#*=}"
      shift
      ;;
    aime24|AIME24|aime2024|AIME2024|all|ALL)
      BENCHMARK="$1"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

BENCHMARK="$(printf '%s' "${BENCHMARK}" | tr '[:upper:]' '[:lower:]')"

ensure_aime24_dataset() {
  if [[ "${INGPO_AUTO_DOWNLOAD_EVAL_DATASETS:-1}" == "0" ]]; then
    [[ -f "${INGPO_ROOT}/data/aime24/dataset_dict.json" ]] || {
      echo "Missing ${INGPO_ROOT}/data/aime24. Run: python scripts/download_eval_datasets.py aime24" >&2
      return 1
    }
    return 0
  fi

  "${INGPO_PYTHON_CMD[@]}" "${INGPO_ROOT}/scripts/download_eval_datasets.py" aime24 --data-dir "${INGPO_ROOT}/data"
}

case "${BENCHMARK}" in
  all|"")
    EXP_NAME="${APP_EXPERIMENT_NAME:-eval-${CFG_NAME}}"
    ;;
  aime24|aime2024)
    ensure_aime24_dataset
    CFG="${CFG},${INGPO_ROOT}/configs/evaluation/aime24_only.jsonnet"
    EXP_NAME="${APP_EXPERIMENT_NAME:-eval-${CFG_NAME}-aime24}"
    ;;
  *)
    echo "Unsupported benchmark '${BENCHMARK}'. Supported values: all, aime24." >&2
    exit 2
    ;;
esac

ingpo_eval "${EXP_NAME}" "${CFG}" "${LAST_POLICY}" "${EXTRA_ARGS[@]}"
