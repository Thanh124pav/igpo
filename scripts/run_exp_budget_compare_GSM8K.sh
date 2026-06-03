#!/usr/bin/env bash
# Matched GSM8K comparison: SPO-tree, GRPO, VinePPO, and budget-allocation tree.
# Defaults:
#   TREE=666
#   MODEL=rho1bSft2
#   N_TV=8

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${INGPO_TREE:-666}}"
MODEL="${MODEL:-rho1bSft2}"
N_TV="${N_TV:-${INGPO_N_TV_ESTIMATES:-8}}"
NUM_ITER="${NUM_ITER:-}"
TAG="${EXP_TAG:-budget-compare-gsm8k-${MODEL}-${TREE}-n${N_TV}}"

EXTRA_ARGS=("$@")
if [[ -n "${NUM_ITER}" ]]; then
  EXTRA_ARGS+=(--override "num_iterations=${NUM_ITER}")
fi

APP_EXPERIMENT_NAME="${TAG}-spo" \
  TREE="${TREE}" INGPO_TREE="${TREE}" \
  bash "${INGPO_ROOT}/scripts/run_baseline.sh" spo_tree_GSM8K "${EXTRA_ARGS[@]}"

APP_EXPERIMENT_NAME="${TAG}-grpo" \
  bash "${INGPO_ROOT}/scripts/run_baseline.sh" grpo_GSM8K "${EXTRA_ARGS[@]}"

APP_EXPERIMENT_NAME="${TAG}-vineppo" \
  bash "${INGPO_ROOT}/scripts/run_baseline.sh" vineppo_GSM8K "${EXTRA_ARGS[@]}"

APP_EXPERIMENT_NAME="${TAG}-budget" \
  TREE="${TREE}" MODEL="${MODEL}" INGPO_N_TV_ESTIMATES="${N_TV}" \
  bash "${INGPO_ROOT}/scripts/train_budget_alloc_tree_GSM8K.sh" "${EXTRA_ARGS[@]}"
