#!/usr/bin/env bash
# Train VinePPO on MATH. Default model: rho1bSft2.
# This mirrors the shipped VinePPO GSM8K entry point and enables MATH comparisons.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-rho1bSft2}"
EXP_NAME="${APP_EXPERIMENT_NAME:-vineppo-${MODEL}-math}"

case "${MODEL}" in
  rho1bSft2)
    CFGS="${INGPO_ROOT}/configs/polIter_rho1bSft2_vineppo_MATH.jsonnet"
    ;;
  *)
    echo "[vineppo_math] unsupported MODEL=${MODEL}; expected rho1bSft2" >&2
    exit 2
    ;;
esac

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
