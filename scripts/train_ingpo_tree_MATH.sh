#!/usr/bin/env bash
# Train InGPO-tree on MATH.
#
# Tree shape via TREE=<digits> (or INGPO_TREE=<digits> for back-compat).
# Any shape works — if the matching branch_factor_<shape>.jsonnet does
# not exist, _common.sh:ensure_tree_config auto-generates one under
# configs/episode_generators/_generated/.
#   TREE=666     -> depth 3, M=600 (default)
#   TREE=6666    -> depth 4, M=500
#   TREE=8888    -> depth 4, M=500 (auto-generated)
#   TREE=3456    -> depth 4, M=500 (auto-generated; mixed widths)
# Override the auto M with TREE_M=<int>.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-qwen1_5b_base}"
TREE="${TREE:-${INGPO_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${TREE}-${MODEL}-math}"

CFGS="$(resolve_math_config ingpo_tree "${MODEL}")"
CFGS+=",$(ensure_tree_config "${TREE}")"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
