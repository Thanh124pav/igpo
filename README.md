# treetune: Unified RL Framework for Reasoning LLMs

Một codebase thống nhất để huấn luyện LLM trên các tác vụ suy luận (MATH, GSM8K, Point24) bằng nhiều thuật toán RL khác nhau. Tất cả các thuật toán đứng ngang hàng, chọn bằng config.

## Algorithm catalog

| Thuật toán | Trainer | Episode generator | Inference strategy | Script |
|------------|---------|-------------------|--------------------|--------|
| **PPO** — vanilla Proximal Policy Optimization | `ppo` | `math_episode_generator` | `cot` | `train_ppo_MATH.sh` |
| **GRPO** — Group Relative PO (DeepSeek) | `ppo` | `math_episode_generator_w_group_advantages` (adv=`grpo`) | `cot` | `train_grpo_MATH.sh` |
| **RLOO** — REINFORCE Leave-One-Out | `ppo` | `math_episode_generator_w_group_advantages` (adv=`rloo`) | `cot` | `train_rloo_GSM8K.sh` |
| **VinePPO** — PPO với vine-style value baseline | `ppo` | `vineppo_episode_generator` | `cot` | `train_vineppo_GSM8K.sh` |
| **DPO** — Direct Preference Optimization (positive variant) | `dpo_positive` | `math_dpo_positive_episode_generator` | `cot` | `train_dpo_MATH.sh` |
| **RestEM** — Rejection sampling + EM-style FT | `restem` | `math_restem_episode_generator` | `cot` | `train_restem_MATH.sh` |
| **SPO-chain** — Segment PO trên chain | `ppo` | `math_episode_generator` | `cot` | `train_spo_chain_MATH.sh` |
| **SPO-tree** — Segment PO trên cây branching | `ppo` | `hybrid_episode_generator` | `hybrid` | `train_spo_tree_MATH.sh` |
| **InGPO** — Information-Gated PO (SPO + ValueShare + Prune) | `ppo` | `ingpo_episode_generator` | `ingpo` | `train_ingpo_tree_MATH.sh` |

Mỗi thuật toán có file canonical ở `configs/algorithms/<algo>.libsonnet` — thin overlay set `(trainer, episode_generator, inference_strategy)` types. Người dùng compose với model/task base để tạo full experiment config.

## Cấu trúc thư mục

```
ingpo/
├── treetune/                         # Python package thống nhất
│   ├── common/                       # registry, FromParams, Params utilities
│   ├── trainers/                     # ppo, dpo_positive, restem, mle, ...
│   ├── episode_generators/           # tất cả episode generators (PPO/GRPO/DPO/RestEM/VinePPO/SPO/InGPO)
│   ├── inference_strategies/         # cot, hybrid, ingpo, ...
│   ├── ingpo/                        # InGPO core helpers (TV bound, budget allocation, local gates)
│   ├── runtime/                      # policy iteration runtime
│   ├── models/, tasks/, analyzers/   # SPO infrastructure
│   └── main.py                       # entry point (treetune.main)
├── guidance/                         # vendored guidance lib (parsing prompts)
├── configs/
│   ├── algorithms/                   # ppo.libsonnet, grpo.libsonnet, ... (9 files)
│   ├── trainers/, episode_generators/, inference_strategies/, models/, tasks/
│   ├── polIter_<model>_<algo>_<dataset>.jsonnet  # full experiment configs
│   ├── ablations/, baselines/        # InGPO-specific overlays
│   ├── ingpo_defaults.libsonnet, ingpo_overlay.libsonnet
│   └── episode_generators/branch_factor_*.jsonnet  # tree shape overlays
├── scripts/                          # train_<algo>_<dataset>.sh + utilities
├── tests/                            # unit tests
├── docs/legacy/                      # legacy SPO README/LICENSE/Dockerfile
└── README.md
```

## Bắt đầu nhanh

### Cài đặt

```bash
bash scripts/setup.sh
```

### Khởi động vLLM server (cần cho scoring)

```bash
bash scripts/start_vllm_server.sh /path/to/model 8000 42 32 0
export APP_OPENAI_VLLM_API_BASE=http://127.0.0.1:8000/v1
```

### Huấn luyện một thuật toán

```bash
# PPO trên MATH với model mặc định
bash scripts/train_ppo_MATH.sh

# GRPO với model khác
MODEL=deepseekR1Qwen bash scripts/train_grpo_MATH.sh

# DPO
bash scripts/train_dpo_MATH.sh

# VinePPO trên GSM8K
bash scripts/train_vineppo_GSM8K.sh

# SPO-tree với tree shape tùy chỉnh
TREE=6666 bash scripts/train_spo_tree_MATH.sh

# InGPO-tree (mặc định 666)
INGPO_TREE=666 bash scripts/train_ingpo_tree_MATH.sh
```

### Đánh giá

```bash
bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH \
    experiments/ingpo-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained

# Chỉ đánh giá AIME 2024
bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH \
    experiments/ingpo-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained \
    --dataset aime24

# Đánh giá nhiều dataset
bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH \
    experiments/ingpo-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained \
    --datasets math,aime24,olympiadbench \
    --debug_mode=true

# Ghép thêm config override; config sau override config trước
bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH \
    experiments/ingpo-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained \
    --config local/math_local_10 \
    --config local/math_local_runtime \
    --dataset math

# Cũng có thể truyền danh sách config phân tách bằng dấu phẩy
bash scripts/evaluate.sh \
    polIter_qwen1_5b_base_ingpo_tree_MATH,local/math_local_10 \
    experiments/ingpo-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained \
    --dataset math

# Đổi model/checkpoint, tokenizer, context và generation limit trực tiếp
bash scripts/evaluate.sh \
    polIter_deepseekR1Qwen_ingpo_tree_MATH \
    HuggingFaceTB/SmolLM2-135M \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --context-length 4096 \
    --max-new-tokens 1024 \
    --dataset aime24

# Eval lần lượt mọi checkpoint trong một experiment
APP_EXPERIMENT_NAME=eval-training-sweep \
bash scripts/evaluate.sh \
    polIter_deepseekR1Qwen_ingpo_tree_MATH \
    experiments/ingpo-tree-666-deepseekR1Qwen-math \
    --all-checkpoints \
    --context-length 4096 \
    --datasets aime24,aime25

# Xem danh sách alias
bash scripts/evaluate.sh --list-datasets
```

Nếu không truyền `--dataset` hoặc `--datasets`, script vẫn chạy toàn bộ
`inference_pipelines` trong config như trước.

## Compose configs

Mỗi config experiment ghép từ các lớp overlay:

```
[gvar.jsonnet]                              # global vars
+ [prompt_library/<task>.jsonnet]           # task-specific prompts
+ [runtimes/policy_iteration.jsonnet]       # runtime
+ [episode_generators/<eg>.jsonnet]         # episode generator type
+ [trainers/<algo>_<dataset>.jsonnet]       # trainer hyper-params
+ [models/<model>.jsonnet]                  # model
+ {custom overrides}
```

Để tạo experiment mới, copy một `polIter_*.jsonnet` đã có rồi đổi các overlay.

## InGPO — chi tiết thuật toán

InGPO hiện dùng `budget_allocation`: TV probes estimate reward variance cho frontier nodes, rồi phân bổ branch budget theo variance.

Các path SHARE/PRUNE còn lại chỉ dùng sibling-local TV comparison; code generate lời giải tham chiếu cũ đã bị loại khỏi production path.

Tham số mặc định (`configs/ingpo_defaults.libsonnet`):

| Tham số | Default | Ý nghĩa |
|---------|---------|---------|
| `m` | 100 | Số continuation anchors tối đa cho sibling-local TV |
| `epsilon` | 0.02 | Ngưỡng value gap cho Prune |
| `local_value_share` | true | Chỉ dùng path local so sánh siblings; global Y path đã bị loại bỏ |
| `demo_examples_per_tree` | 2 | Số demo budget/local gate lưu mỗi cây |
| `budget_lambda` | 0.02 | Ngưỡng trong trọng số `sqrt(sigma_i^4 - budget_lambda)` |
| `n_min` | 0 | Số nhánh trả về khi `sigma_i^4 - budget_lambda < 0` |
| `skip_near_leaf_expand` | false | Budget allocation: ở depth cuối, bỏ TV/budget allocation và expand uniform theo branch factor B |
| `root_allocation` | false | Budget allocation: estimate variance ở các root trong minibatch và phân bổ branch budget depth 0 giữa các root đó |

Override trong file `.jsonnet`:

```jsonnet
{ ingpo+: { epsilon: 0.05, K: 20 } }
```

Ablations & sweep nằm trong `configs/ablations/`, `configs/baselines/`, `configs/num_iter_sweep.libsonnet`.

## Logging offline (no internet)

InGPO ghi mọi metric ra file để dùng offline:

- `<exp>/training_timing.jsonl` — mỗi iteration 1 dòng: `train_total_seconds` (không gồm eval), `eval_seconds`, cumulative wall.
- `<exp>/ingpo_demos/demos.jsonl` — mỗi tree: stats, per_depth, tree_construction_seconds, budget/local-gate demos.
- `<exp>/ingpo_demos/demos.md` — bản Markdown human-readable.

Xem live:

```bash
bash scripts/tail_demos.sh <exp_name>             # markdown
bash scripts/tail_demos.sh <exp_name> jsonl       # jsonl
python scripts/inspect_demos.py <exp>/ingpo_demos/demos.jsonl --summary
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q       # unit tests (no GPU needed)
bash scripts/run_smoke.sh                     # config compile + unit tests
bash scripts/train_debug.sh                   # E2E 2 iterations, depth 2 (needs vLLM)
```

## Migration notes

Phiên bản trước có hai layer riêng: `ingpo/spo/` (SPO) + `ingpo/ingpo_src/` (InGPO ext). Refactor này gộp tất cả vào `treetune/` ở top level:

| Cũ | Mới |
|----|-----|
| `ingpo/spo/src/treetune/` | `ingpo/treetune/` |
| `ingpo/spo/src/guidance/` | `ingpo/guidance/` |
| `ingpo/ingpo_src/ingpo_ext/core/` | `ingpo/treetune/ingpo/` |
| `ingpo/ingpo_src/ingpo_ext/episode_generators/ingpo_episode_generator.py` | `ingpo/treetune/episode_generators/ingpo_episode_generator.py` |
| `ingpo/ingpo_src/ingpo_ext/inference_strategies/ingpo_inference_strategy.py` | `ingpo/treetune/inference_strategies/ingpo_inference_strategy.py` |
| `ingpo/spo/configs/*` + `ingpo/configs/*` | `ingpo/configs/*` (merged) |
| `ingpo/spo/scripts/*` + `ingpo/scripts/*` | `ingpo/scripts/*` (merged) |
| `ingpo_main.py` shim | xoá — dùng `python -m treetune.main` |
| `import ingpo_ext.X` | `import treetune.ingpo.X` |
| `setup.py` (ingpo_ext) | `setup.py` (treetune, single package) |

DAPO không có trong scope — dùng GRPO (`scripts/train_grpo_MATH.sh`) thay thế.

## License

Code base treetune kế thừa MIT License của SPO. Xem `docs/legacy/LICENSE_SPO`.

## Tham khảo

- [SPO paper](https://github.com/AIFrameResearch/SPO) — Segment Policy Optimization gốc
- [GRPO paper](https://arxiv.org/abs/2402.03300) — DeepSeek-Math
- [VinePPO paper](https://arxiv.org/abs/2410.01679) — McGill
- [DPO paper](https://arxiv.org/abs/2305.18290) — Stanford
- `PLAN.md` — đặc tả chi tiết thuật toán InGPO

### Benchmark được đánh giá trong lúc training

Chuẩn bị các dataset eval local:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deeplearning
python scripts/download_eval_datasets.py
```

Các cấu hình training trên MATH chạy thêm năm benchmark ở mỗi lần evaluation:

- `aime24_test`: `math-ai/aime24` (30 bài).
- `aime25_test`: `math-ai/aime25` (30 bài).
- `amc23_test`: `math-ai/amc23` (40 bài AMC 2023).
- `olympiadbench_test`: file
  `OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet` của
  `Hothan/OlympiadBench` (674 bài).
- `collegeMath_test`: `data/collegeMath` (2,818 bài).

Downloader pin revision Hugging Face và lưu từng nguồn ở dạng `DatasetDict` dưới
`data/`. Thư mục cũ `data/olympiadbench` có 675 bài, gồm một bài không còn trong
bản Hugging Face hiện tại và 15 đề còn placeholder lỗi; benchmark dùng bản chuẩn
ở `data/olympiadbench_hf`.

Smoke test format prompt và generation bằng model nhỏ:

```bash
python scripts/smoke_eval_datasets.py --local-files-only
```

Tần suất evaluation vẫn được điều khiển bởi `INGPO_EVAL_EVERY_N_ITERATIONS`.
