#!/usr/bin/env bash
# Train simulation-lemma budget-allocation tree on GSM8K.
# Matched to shipped SPO/GRPO/VinePPO GSM8K baselines by default.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-rho1bSft2}"
TREE="${TREE:-${INGPO_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-budget-alloc-tree-${TREE}-${MODEL}-gsm8k}"

case "${MODEL}" in
  rho1bSft2)
    CFGS="${INGPO_ROOT}/configs/polIter_rho1bSft2_budget_alloc_tree_GSM8K.jsonnet"
    ;;
  qwen05b)
    CFGS="${INGPO_ROOT}/configs/polIter_qwen05b_budget_alloc_tree_GSM8K.jsonnet"
    ;;
  *)
    echo "[budget_alloc_gsm8k] unsupported MODEL=${MODEL}; expected rho1bSft2 or qwen05b" >&2
    exit 2
    ;;
esac
CFGS+=",$(ensure_tree_config "${TREE}")"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
