#!/usr/bin/env bash
# Evaluate a trained checkpoint, optionally selecting inference pipelines.
# Usage:
#   bash scripts/evaluate.sh <config[,override...]> <model_or_checkpoint> \
#     [--config <override>]... \
#     [--dataset <name>]... [--datasets <name1,name2>] [extra args...]
#
# Examples:
#   # All pipelines configured by the alias (backwards-compatible default).
#   bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH <checkpoint>
#
#   # One dataset.
#   bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH \
#     <checkpoint> --dataset aime24
#
#   # Multiple datasets plus normal treetune arguments.
#   bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH \
#     <checkpoint> --datasets math,aime24,olympiadbench --debug_mode=true

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate.sh <config[,override...]> [model_or_checkpoint] \
    [options] [treetune args]

Config options:
  --config CONFIG         Append a Jsonnet config/overlay. Repeat as needed.

Configs are merged from left to right. Each config may be:
  - an alias under configs/, without .jsonnet
  - a path relative to configs/
  - a repository-relative or absolute .jsonnet path

Runtime override options:
  --tokenizer MODEL       Override tokenizer for the runtime and all pipelines.
  --context-length N      Set pipeline context size and vLLM max model length.
  --max-new-tokens N      Override the maximum generated tokens per response.
  --num-samples N        Override generated samples per problem for pass@n metrics.
  --temperature FLOAT    Override sampling temperature for eval generation.
  --top-p FLOAT          Override nucleus sampling top-p for eval generation.

Checkpoint options:
  model_or_checkpoint    Optional when --checkpoint, --checkpoint-glob, or
                          --checkpoint-template supplies checkpoint(s).
  --checkpoint PATH       Append another model/checkpoint. Repeat as needed.
  --checkpoint-glob GLOB  Append checkpoints matching a quoted shell glob.
  --checkpoint-template TEMPLATE
                          Append rendered checkpoints; {...} expressions may use it.
  --checkpoint-iters SPEC Iterations for --checkpoint-template: START:END[:STEP] or A,B,C.
  --all-checkpoints       Treat the positional path as an experiment directory
                          and evaluate checkpoints/*/hf_pretrained in order.

Dataset options:
  --dataset NAME [...]    Evaluate one or more pipelines.
  --datasets A,B,C        Evaluate a comma-separated list of pipelines.
  --list-datasets         Print supported aliases and exit.

Known aliases:
  math, gsm8k, aime24, aime25, amc23, olympiadbench, college_math, point24

Raw inference pipeline names such as math_test or aime24_test are also accepted.
With no dataset option, every pipeline from the selected config is evaluated.
EOF
}

list_datasets() {
  cat <<'EOF'
math             -> math_test
gsm8k            -> gsm8k_test
aime24           -> aime24_test
aime25           -> aime25_test
amc23            -> amc23_test
olympiadbench     -> olympiadbench_test
college_math     -> collegeMath_test
point24          -> point24_test
EOF
}

normalize_pipeline_name() {
  local name="$1"
  case "${name,,}" in
    math|math_test) echo "math_test" ;;
    gsm8k|gsm8k_test) echo "gsm8k_test" ;;
    aime24|aime24_test) echo "aime24_test" ;;
    aime25|aime25_test) echo "aime25_test" ;;
    amc23|amc23_test) echo "amc23_test" ;;
    olympiadbench|olympiadbench_hf|olympiadbench_test)
      echo "olympiadbench_test"
      ;;
    college_math|college-math|collegemath|collegemath_test)
      echo "collegeMath_test"
      ;;
    point24|point24_test) echo "point24_test" ;;
    all) echo "all" ;;
    *)
      if [[ "${name}" =~ ^[A-Za-z0-9_]+$ ]]; then
        echo "${name}"
      else
        echo "Invalid dataset or pipeline name: ${name}" >&2
        return 2
      fi
      ;;
  esac
}

resolve_config() {
  local config="$1"
  local candidate

  [[ -n "${config}" ]] || {
    echo "Config name cannot be empty" >&2
    return 2
  }

  if [[ -f "${config}" ]]; then
    realpath "${config}"
    return
  fi

  if [[ -f "${INGPO_ROOT}/${config}" ]]; then
    realpath "${INGPO_ROOT}/${config}"
    return
  fi

  candidate="${INGPO_ROOT}/configs/${config}"
  if [[ "${candidate}" != *.jsonnet ]]; then
    candidate+=".jsonnet"
  fi
  if [[ -f "${candidate}" ]]; then
    realpath "${candidate}"
    return
  fi

  echo "Cannot find config ${config}" >&2
  return 2
}

require_positive_integer() {
  local option="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${option} requires a positive integer, got: ${value}" >&2
    return 2
  fi
}

checkpoint_label() {
  local checkpoint="${1%/}"
  local label
  label="$(basename "${checkpoint}")"
  if [[ "${label}" == "hf_pretrained" ]]; then
    label="$(basename "$(dirname "${checkpoint}")")"
  fi
  printf '%s' "${label}" | tr -c 'A-Za-z0-9_.-' '-'
}

checkpoint_iteration() {
  local checkpoint="${1%/}"
  local label
  label="$(checkpoint_label "${checkpoint}")"
  if [[ "${label}" =~ ckpt--iter_0*([0-9]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf '0'
  fi
}

resolve_checkpoint_path() {
  local checkpoint="$1"
  if [[ -d "${checkpoint}" && ! -f "${checkpoint}/config.json" && -d "${checkpoint}/hf_pretrained" ]]; then
    realpath "${checkpoint}/hf_pretrained"
  elif [[ -e "${checkpoint}" ]]; then
    realpath "${checkpoint}"
  else
    # Keep Hugging Face model IDs unchanged.
    echo "${checkpoint}"
  fi
}

expand_checkpoint_template() {
  local template="$1"
  local iter_spec="$2"
  [[ -n "${iter_spec}" ]] || {
    echo "--checkpoint-template requires --checkpoint-iters" >&2
    return 2
  }
  python - "$template" "$iter_spec" <<'PYCHECKPOINT'
import ast
import operator
import re
import sys

template, spec = sys.argv[1], sys.argv[2]

def parse_iters(value):
    if ":" in value:
        parts = value.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("--checkpoint-iters range must be START:END[:STEP]")
        start, end = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if step == 0:
            raise ValueError("--checkpoint-iters STEP cannot be zero")
        return range(start, end + (1 if step > 0 else -1), step)
    return [int(part) for part in value.split(",") if part]

ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.FloorDiv: operator.floordiv, ast.Div: operator.floordiv,
       ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg}
cmps = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
        ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}

def eval_expr(expr, it):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
            return node.value
        if isinstance(node, ast.Name) and node.id == "it":
            return it
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in cmps:
            return cmps[type(node.ops[0])](visit(node.left), visit(node.comparators[0]))
        raise ValueError(f"unsupported expression in checkpoint template: {expr}")
    return visit(ast.parse(expr, mode="eval"))

pattern = re.compile(r"\{([^{}]+)\}")
for it in parse_iters(spec):
    rendered_values = [eval_expr(match.group(1), it) for match in pattern.finditer(template)]
    if any(isinstance(value, bool) for value in rendered_values) and not all(rendered_values):
        continue

    def render(match):
        value = eval_expr(match.group(1), it)
        return str(it) if isinstance(value, bool) else str(value)

    print(pattern.sub(render, template))
PYCHECKPOINT
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--list-datasets" ]]; then
  list_datasets
  exit 0
fi

CONFIG_SPEC="${1:?$(usage >&2)}"
shift
LAST_POLICY=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  LAST_POLICY="$1"
  shift
fi

CONFIG_PATHS=()
SELECTED_PIPELINES=()
PASSTHROUGH_ARGS=()
EXTRA_CHECKPOINTS=()
CHECKPOINT_GLOBS=()
CHECKPOINT_TEMPLATES=()
CHECKPOINT_ITERS=""
EVAL_TOKENIZER=""
EVAL_CONTEXT_LENGTH=""
EVAL_MAX_NEW_TOKENS=""
EVAL_NUM_SAMPLES=""
EVAL_TEMPERATURE=""
EVAL_TOP_P=""
ALL_CHECKPOINTS=0

IFS=',' read -r -a initial_configs <<< "${CONFIG_SPEC}"
for config in "${initial_configs[@]}"; do
  CONFIG_PATHS+=("$(resolve_config "${config}")")
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a config" >&2; exit 2; }
      CONFIG_PATHS+=("$(resolve_config "$2")")
      shift 2
      ;;
    --config=*)
      CONFIG_PATHS+=("$(resolve_config "${1#--config=}")")
      shift
      ;;
    --tokenizer)
      [[ $# -ge 2 ]] || { echo "--tokenizer requires a model or path" >&2; exit 2; }
      EVAL_TOKENIZER="$2"
      shift 2
      ;;
    --tokenizer=*)
      EVAL_TOKENIZER="${1#--tokenizer=}"
      shift
      ;;
    --context-length)
      [[ $# -ge 2 ]] || { echo "--context-length requires a value" >&2; exit 2; }
      require_positive_integer "--context-length" "$2"
      EVAL_CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --context-length=*)
      EVAL_CONTEXT_LENGTH="${1#--context-length=}"
      require_positive_integer "--context-length" "${EVAL_CONTEXT_LENGTH}"
      shift
      ;;
    --max-new-tokens)
      [[ $# -ge 2 ]] || { echo "--max-new-tokens requires a value" >&2; exit 2; }
      require_positive_integer "--max-new-tokens" "$2"
      EVAL_MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --max-new-tokens=*)
      EVAL_MAX_NEW_TOKENS="${1#--max-new-tokens=}"
      require_positive_integer "--max-new-tokens" "${EVAL_MAX_NEW_TOKENS}"
      shift
      ;;
    --num-samples)
      [[ $# -ge 2 ]] || { echo "--num-samples requires a value" >&2; exit 2; }
      require_positive_integer "--num-samples" "$2"
      EVAL_NUM_SAMPLES="$2"
      shift 2
      ;;
    --num-samples=*)
      EVAL_NUM_SAMPLES="${1#--num-samples=}"
      require_positive_integer "--num-samples" "${EVAL_NUM_SAMPLES}"
      shift
      ;;
    --temperature)
      [[ $# -ge 2 ]] || { echo "--temperature requires a value" >&2; exit 2; }
      EVAL_TEMPERATURE="$2"
      shift 2
      ;;
    --temperature=*)
      EVAL_TEMPERATURE="${1#--temperature=}"
      shift
      ;;
    --top-p)
      [[ $# -ge 2 ]] || { echo "--top-p requires a value" >&2; exit 2; }
      EVAL_TOP_P="$2"
      shift 2
      ;;
    --top-p=*)
      EVAL_TOP_P="${1#--top-p=}"
      shift
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "--checkpoint requires a path or model ID" >&2; exit 2; }
      EXTRA_CHECKPOINTS+=("$2")
      shift 2
      ;;
    --checkpoint=*)
      EXTRA_CHECKPOINTS+=("${1#--checkpoint=}")
      shift
      ;;
    --checkpoint-glob)
      [[ $# -ge 2 ]] || { echo "--checkpoint-glob requires a quoted glob" >&2; exit 2; }
      CHECKPOINT_GLOBS+=("$2")
      shift 2
      ;;
    --checkpoint-glob=*)
      CHECKPOINT_GLOBS+=("${1#--checkpoint-glob=}")
      shift
      ;;
    --checkpoint-template)
      [[ $# -ge 2 ]] || { echo "--checkpoint-template requires a template" >&2; exit 2; }
      CHECKPOINT_TEMPLATES+=("$2")
      shift 2
      ;;
    --checkpoint-template=*)
      CHECKPOINT_TEMPLATES+=("${1#--checkpoint-template=}")
      shift
      ;;
    --checkpoint-iters)
      [[ $# -ge 2 ]] || { echo "--checkpoint-iters requires START:END[:STEP] or a comma list" >&2; exit 2; }
      CHECKPOINT_ITERS="$2"
      shift 2
      ;;
    --checkpoint-iters=*)
      CHECKPOINT_ITERS="${1#--checkpoint-iters=}"
      shift
      ;;
    --all-checkpoints)
      ALL_CHECKPOINTS=1
      shift
      ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "--dataset requires a name" >&2; exit 2; }
      shift
      dataset_count=0
      while [[ $# -gt 0 && "$1" != -* ]]; do
        SELECTED_PIPELINES+=("$(normalize_pipeline_name "$1")")
        dataset_count=$((dataset_count + 1))
        shift
      done
      [[ ${dataset_count} -gt 0 ]] || {
        echo "--dataset requires at least one name" >&2
        exit 2
      }
      ;;
    --dataset=*)
      SELECTED_PIPELINES+=(
        "$(normalize_pipeline_name "${1#--dataset=}")"
      )
      shift
      ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "--datasets requires a comma-separated list" >&2; exit 2; }
      IFS=',' read -r -a dataset_names <<< "$2"
      for dataset_name in "${dataset_names[@]}"; do
        [[ -n "${dataset_name}" ]] || continue
        SELECTED_PIPELINES+=("$(normalize_pipeline_name "${dataset_name}")")
      done
      shift 2
      ;;
    --datasets=*)
      IFS=',' read -r -a dataset_names <<< "${1#--datasets=}"
      for dataset_name in "${dataset_names[@]}"; do
        [[ -n "${dataset_name}" ]] || continue
        SELECTED_PIPELINES+=("$(normalize_pipeline_name "${dataset_name}")")
      done
      shift
      ;;
    --list-datasets)
      list_datasets
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

CFG="$(IFS=,; echo "${CONFIG_PATHS[*]}")"

if [[ -n "${EVAL_TOKENIZER}" || -n "${EVAL_CONTEXT_LENGTH}" || -n "${EVAL_MAX_NEW_TOKENS}" || -n "${EVAL_NUM_SAMPLES}" || -n "${EVAL_TEMPERATURE}" || -n "${EVAL_TOP_P}" ]]; then
  export APP_EVAL_TOKENIZER="${EVAL_TOKENIZER}"
  export APP_EVAL_CONTEXT_LENGTH="${EVAL_CONTEXT_LENGTH}"
  export APP_EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}"
  export APP_EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES}"
  export APP_EVAL_TEMPERATURE="${EVAL_TEMPERATURE}"
  export APP_EVAL_TOP_P="${EVAL_TOP_P}"
  CFG+=",${INGPO_ROOT}/configs/evaluation/cli_overrides.jsonnet"
  echo "Evaluation overrides:"
  [[ -n "${EVAL_TOKENIZER}" ]] && echo "  tokenizer=${EVAL_TOKENIZER}"
  [[ -n "${EVAL_CONTEXT_LENGTH}" ]] && echo "  context_length=${EVAL_CONTEXT_LENGTH}"
  [[ -n "${EVAL_MAX_NEW_TOKENS}" ]] && echo "  max_new_tokens=${EVAL_MAX_NEW_TOKENS}"
  [[ -n "${EVAL_NUM_SAMPLES}" ]] && echo "  num_samples=${EVAL_NUM_SAMPLES}"
  [[ -n "${EVAL_TEMPERATURE}" ]] && echo "  temperature=${EVAL_TEMPERATURE}"
  [[ -n "${EVAL_TOP_P}" ]] && echo "  top_p=${EVAL_TOP_P}"
fi

if [[ ${#SELECTED_PIPELINES[@]} -gt 0 ]]; then
  UNIQUE_PIPELINES=()
  for pipeline in "${SELECTED_PIPELINES[@]}"; do
    if [[ "${pipeline}" == "all" ]]; then
      UNIQUE_PIPELINES=()
      break
    fi
    if [[ ! " ${UNIQUE_PIPELINES[*]} " =~ " ${pipeline} " ]]; then
      UNIQUE_PIPELINES+=("${pipeline}")
    fi
  done

  if [[ ${#UNIQUE_PIPELINES[@]} -gt 0 ]]; then
    APP_EVAL_PIPELINES="$(IFS=,; echo "${UNIQUE_PIPELINES[*]}")"
    export APP_EVAL_PIPELINES
    CFG+=",${INGPO_ROOT}/configs/evaluation/select_pipelines.jsonnet"
    echo "Selected evaluation pipelines: ${APP_EVAL_PIPELINES}"
  fi
fi

BASE_CONFIG_NAME="$(basename "${CONFIG_PATHS[0]}" .jsonnet)"
BASE_EXP_NAME="${APP_EXPERIMENT_NAME:-eval-${BASE_CONFIG_NAME}}"
CHECKPOINTS=()

if [[ "${ALL_CHECKPOINTS}" == "1" ]]; then
  [[ -n "${LAST_POLICY}" ]] || {
    echo "--all-checkpoints requires a positional experiment directory" >&2
    exit 2
  }
  CHECKPOINT_ROOT="${LAST_POLICY%/}"
  if [[ -d "${CHECKPOINT_ROOT}/checkpoints" ]]; then
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/checkpoints"
  fi
  [[ -d "${CHECKPOINT_ROOT}" ]] || {
    echo "Checkpoint directory does not exist: ${CHECKPOINT_ROOT}" >&2
    exit 2
  }
  mapfile -t CHECKPOINTS < <(
    find "${CHECKPOINT_ROOT}" \
      -mindepth 2 \
      -maxdepth 2 \
      -type d \
      -name hf_pretrained \
      -print | sort -V
  )
  [[ ${#CHECKPOINTS[@]} -gt 0 ]] || {
    echo "No checkpoints/*/hf_pretrained found under ${CHECKPOINT_ROOT}" >&2
    exit 2
  }
else
  if [[ -n "${LAST_POLICY}" ]]; then
    CHECKPOINTS+=("${LAST_POLICY}")
  fi
fi

CHECKPOINTS+=("${EXTRA_CHECKPOINTS[@]}")
for checkpoint_glob in "${CHECKPOINT_GLOBS[@]}"; do
  mapfile -t glob_matches < <(compgen -G "${checkpoint_glob}" | sort -V || true)
  [[ ${#glob_matches[@]} -gt 0 ]] || {
    echo "No checkpoints matched glob: ${checkpoint_glob}" >&2
    exit 2
  }
  CHECKPOINTS+=("${glob_matches[@]}")
done
for checkpoint_template in "${CHECKPOINT_TEMPLATES[@]}"; do
  mapfile -t template_matches < <(expand_checkpoint_template "${checkpoint_template}" "${CHECKPOINT_ITERS}")
  [[ ${#template_matches[@]} -gt 0 ]] || {
    echo "No checkpoints rendered from template: ${checkpoint_template}" >&2
    exit 2
  }
  CHECKPOINTS+=("${template_matches[@]}")
done

[[ ${#CHECKPOINTS[@]} -gt 0 ]] || {
  echo "No checkpoint/model supplied. Provide model_or_checkpoint, --checkpoint, --checkpoint-glob, or --checkpoint-template." >&2
  exit 2
}

for index in "${!CHECKPOINTS[@]}"; do
  checkpoint="$(resolve_checkpoint_path "${CHECKPOINTS[$index]}")"
  label="$(checkpoint_label "${checkpoint}")"
  iteration="$(checkpoint_iteration "${checkpoint}")"
  if [[ ${#CHECKPOINTS[@]} -eq 1 ]]; then
    exp_name="${BASE_EXP_NAME}"
  else
    printf -v run_number '%03d' "$((index + 1))"
    exp_name="${BASE_EXP_NAME}-${run_number}-${label}"
  fi

  echo "Evaluating checkpoint $((index + 1))/${#CHECKPOINTS[@]}:"
  echo "  model=${checkpoint}"
  echo "  iteration=${iteration}"
  echo "  experiment=${exp_name}"
  ingpo_eval \
    "${exp_name}" \
    "${CFG}" \
    "${checkpoint}" \
    "${iteration}" \
    "${PASSTHROUGH_ARGS[@]}"
done
