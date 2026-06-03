#!/usr/bin/env bash
# Matched MATH comparison: SPO-tree, GRPO, VinePPO, and budget-allocation tree.
#
# Defaults are chosen so all baselines have shipped configs:
#   TREE=666
#   MODEL=rho1bSft2
#   N_TV=8
#
# Example:
#   TREE=666 MODEL=rho1bSft2 NUM_ITER=10 bash scripts/run_exp_budget_compare_MATH.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${INGPO_TREE:-666}}"
MODEL="${MODEL:-rho1bSft2}"
N_TV="${N_TV:-${INGPO_N_TV_ESTIMATES:-8}}"
NUM_ITER="${NUM_ITER:-}"
TAG="${EXP_TAG:-budget-compare-math-${MODEL}-${TREE}-n${N_TV}}"

EXTRA_ARGS=("$@")
if [[ -n "${NUM_ITER}" ]]; then
  EXTRA_ARGS+=(--override "num_iterations=${NUM_ITER}")
fi

APP_EXPERIMENT_NAME="${TAG}-spo" \
  TREE="${TREE}" MODEL="${MODEL}" \
  bash "${INGPO_ROOT}/scripts/train_spo_tree_MATH.sh" "${EXTRA_ARGS[@]}"

APP_EXPERIMENT_NAME="${TAG}-grpo" \
  MODEL="${MODEL}" \
  bash "${INGPO_ROOT}/scripts/train_grpo_MATH.sh" "${EXTRA_ARGS[@]}"

APP_EXPERIMENT_NAME="${TAG}-vineppo" \
  MODEL="rho1bSft2" \
  bash "${INGPO_ROOT}/scripts/train_vineppo_MATH.sh" "${EXTRA_ARGS[@]}"

APP_EXPERIMENT_NAME="${TAG}-budget" \
  TREE="${TREE}" MODEL="${MODEL}" INGPO_N_TV_ESTIMATES="${N_TV}" \
  bash "${INGPO_ROOT}/scripts/train_budget_alloc_tree_MATH.sh" "${EXTRA_ARGS[@]}"
