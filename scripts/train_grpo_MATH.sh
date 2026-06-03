#!/usr/bin/env bash
# Train GRPO on MATH.  Default model: rho1bSft2.
# Override base with MODEL={deepseekR1Qwen,rho1bSft2,deepseekSft2,qwen1_5b_base}.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-rho1bSft2}"
EXP_NAME="${APP_EXPERIMENT_NAME:-grpo-${MODEL}-math}"

CFGS="$(resolve_math_config grpo "${MODEL}")"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
