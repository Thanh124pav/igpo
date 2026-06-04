#!/usr/bin/env bash
# Smoke test for the simulation-lemma budget-allocation path on tiny models.
#
# Default mode compiles Jsonnet stacks and asserts that the algorithm knobs still
# match the budget-allocation contract.  To launch actual one-iteration runs,
# set RUN_E2E=1.  Example:
#
#   bash scripts/run_budget_alloc_small_model_smoke.sh
#   RUN_E2E=1 INGPO_GPU=0 bash scripts/run_budget_alloc_small_model_smoke.sh
#
# Override model aliases with:
#   SMOKE_MODELS="gpt2 qwen2_0_5b qwen3_0_6b" bash scripts/run_budget_alloc_small_model_smoke.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BASE_CONFIG_REL="${SMOKE_BASE_CONFIG_REL:-polIter_qwen1_5b_base_budget_alloc_tree_MATH.jsonnet}"
SMOKE_CONFIG_REL="${SMOKE_CONFIG_REL:-debug_budget_allocation_smoke.jsonnet}"
SMOKE_MODELS="${SMOKE_MODELS:-gpt2 qwen2_0_5b qwen3_0_6b}"
RUN_E2E="${RUN_E2E:-0}"

export APP_LOG_BACKEND="${APP_LOG_BACKEND:-tensorboard}"

compile_and_check_configs() {
  python3 - "${INGPO_ROOT}" "${BASE_CONFIG_REL}" "${SMOKE_CONFIG_REL}" "${SMOKE_MODELS}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

if importlib.util.find_spec("_jsonnet") is None:
    raise SystemExit(
        "Missing python jsonnet binding `_jsonnet`; install repo requirements before running this smoke test."
    )

import _jsonnet

root = Path(sys.argv[1])
base_config_rel = sys.argv[2]
smoke_config_rel = sys.argv[3]
models = sys.argv[4].split()

jpathdir = [str(root / "configs"), str(root)]
ext_vars = {
    "APP_SEED": "42",
    "APP_OPENAI_VLLM_API_BASE": "http://127.0.0.1:8000/v1",
}

failures = []
for model in models:
    override = root / "configs" / "model_overrides" / f"{model}.jsonnet"
    if not override.exists():
        failures.append(f"{model}: missing model override {override}")
        continue

    snippet = (
        f"(import '{base_config_rel}')"
        f" + (import 'model_overrides/{model}.jsonnet')"
        f" + (import '{smoke_config_rel}')"
    )
    try:
        rendered = _jsonnet.evaluate_snippet(
            f"budget_alloc_small_model_smoke_{model}",
            snippet,
            jpathdir=jpathdir,
            ext_vars=ext_vars,
        )
        cfg = json.loads(rendered)
        eg = cfg["episode_generator"]
        inference = eg["inference_strategy"]
        trainer_args = cfg["trainer"]["general_training_args"]

        checks = {
            "episode generator is InGPO": eg["type"] == "ingpo_episode_generator",
            "inference strategy is InGPO": inference["type"] == "ingpo",
            "algorithm mode remains budget_allocation": inference["ingpo_algorithm_mode"] == "budget_allocation",
            "flexible overhead remains default": inference["ingpo_budget_overhead_mode"] == "flexible",
            "subnode TV estimator remains active": inference["ingpo_tv_estimator"] == "subnode",
            "small n_tv_estimates for smoke": inference["ingpo_n_tv_estimates"] == 4,
            "tree_shape smoke override present": inference["branch_factor_strategy"]["tree_shape"] == [2, 4],
            "tiny train batch for smoke": trainer_args["target_train_batch_size"] == 2,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            failures.append(f"{model}: failed checks: {', '.join(bad)}")
            continue

        print(
            "OK",
            model,
            "hf_model=",
            eg["initial_model_name_or_path"],
            "algorithm=",
            inference["ingpo_algorithm_mode"],
            "tree_shape=",
            inference["branch_factor_strategy"]["tree_shape"],
        )
    except Exception as exc:  # noqa: BLE001 - print all smoke failures together.
        failures.append(f"{model}: {exc}")

if failures:
    print("Small-model budget-allocation smoke failed:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(1)
PY
}

compile_and_check_configs

if [[ "${RUN_E2E}" != "1" ]]; then
  echo "[smoke] Compile/contract checks passed. Set RUN_E2E=1 to launch real tiny runs."
  exit 0
fi

for model in ${SMOKE_MODELS}; do
  exp_name="budget-alloc-smoke-${model}-$(date +%Y%m%d-%H%M%S)"
  cfgs="${INGPO_ROOT}/configs/${BASE_CONFIG_REL},${INGPO_ROOT}/configs/model_overrides/${model}.jsonnet,${INGPO_ROOT}/configs/${SMOKE_CONFIG_REL}"
  echo "[smoke] launching ${model}: ${exp_name}"
  ingpo_run "${exp_name}" "${cfgs}" "$@"
done
