#!/usr/bin/env bash
# Evaluate a trained checkpoint on the MATH/GSM8K test set, mirroring
# `scripts/evaluate_long_cot.sh` from SPO.
# Usage:
#   bash scripts/evaluate.sh <config_alias> <last_policy_path> [extra args...]
#
# Examples:
#   bash scripts/evaluate.sh polIter_deepseekR1Qwen_ingpo_tree_MATH "${INGPO_ROOT}/experiments/exp1-ingpo-tree-666-deepseek-r1-qwen-math/iteration_0010"
#   bash scripts/evaluate.sh polIter_rho1bSft2_ingpo_tree_GSM8K  "<hf-checkpoint>"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CFG_NAME="${1:?Usage: evaluate.sh <config_alias> <last_policy_path>}"
LAST_POLICY="${2:?missing last_policy_path}"
shift 2 || true

CFG="${INGPO_ROOT}/configs/${CFG_NAME}.jsonnet"
[[ -f "${CFG}" ]] || { echo "Cannot find config ${CFG_NAME}"; exit 2; }

EXP_NAME="${APP_EXPERIMENT_NAME:-eval-${CFG_NAME}}"
ingpo_eval "${EXP_NAME}" "${CFG}" "${LAST_POLICY}" "$@"
