#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT="${1:?Usage: bash run_aime24_eval.sh <checkpoint_path_or_hf_model> [extra evaluate args...]}"
shift || true

CONFIG_ALIAS="${CONFIG_ALIAS:-qwen1_5b_base_for_MATH_eval}"

INGPO_GPU="${INGPO_GPU:-0}" \
APP_EXPERIMENT_NAME="${APP_EXPERIMENT_NAME:-eval-aime24}" \
bash "${ROOT_DIR}/scripts/evaluate.sh" "${CONFIG_ALIAS}" "${CHECKPOINT}" --benchmark aime24 "$@"
