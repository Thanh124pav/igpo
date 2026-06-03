#!/usr/bin/env bash
# Train simulation-lemma budget-allocation tree on MATH.
# Matched to SPO-tree by MODEL and TREE so results can be compared directly.
# Override:
#   MODEL={qwen1_5b_base,deepseekR1Qwen,rho1bSft2}
#   TREE=666
#   INGPO_N_TV_ESTIMATES={4,8,16}

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-qwen1_5b_base}"
TREE="${TREE:-${INGPO_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-budget-alloc-tree-${TREE}-${MODEL}-math}"

CFGS="$(resolve_math_config budget_alloc_tree "${MODEL}")"
CFGS+=",$(ensure_tree_config "${TREE}")"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
