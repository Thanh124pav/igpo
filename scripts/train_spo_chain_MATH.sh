#!/usr/bin/env bash
# Train SPO-chain on MATH.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-qwen1b}"
EXP_NAME="${APP_EXPERIMENT_NAME:-spo-chain-${MODEL}-math}"

CFGS="$(resolve_math_config spo_chain "${MODEL}")"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
