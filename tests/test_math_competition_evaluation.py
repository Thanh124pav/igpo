import json
from pathlib import Path

import _jsonnet
import pytest

from treetune.common import Params
from treetune.tasks import Task


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"


def test_math_eval_configs_include_shared_benchmarks():
    eval_configs = (
        "deepseekR1Qwen_for_MATH_eval.jsonnet",
        "qwen1_5b_base_for_MATH_eval.jsonnet",
        "sft_deepseekmath_for_MATH_eval.jsonnet",
        "sft_rho1b_for_MATH_eval.jsonnet",
    )

    for config_name in eval_configs:
        config = (CONFIGS / config_name).read_text()
        assert "evaluation/math_benchmarks.libsonnet" in config
        assert "] + math_benchmark_pipelines" in config


def test_math_benchmark_overlay_defines_all_requested_evals():
    overlay = (CONFIGS / "evaluation" / "math_benchmarks.libsonnet").read_text()

    for inference_name in (
        "aime24_test",
        "aime25_test",
        "amc23_test",
        "olympiadbench_test",
        "collegeMath_test",
    ):
        assert f"inference_name: '{inference_name}'" in overlay


def test_downloaded_tasks_use_local_normalized_math_dataset_fields():
    expected = {
        "aime24_inplace_no_answer_prefix.jsonnet": (
            "'data/aime24'",
            "'problem'",
            "null",
            "'solution'",
        ),
        "aime25_inplace_no_answer_prefix.jsonnet": (
            "'data/aime25'",
            "'problem'",
            "'answer'",
            "null",
        ),
        "amc23_inplace_no_answer_prefix.jsonnet": (
            "'data/amc23'",
            "'question'",
            "'answer'",
            "null",
        ),
        "olympiadbench_hf_inplace_no_answer_prefix.jsonnet": (
            "'data/olympiadbench_hf'",
            "'question'",
            "'final_answer'",
            "'solution'",
        ),
    }

    for config_name, fields in expected.items():
        dataset_path, problem_field, answer_field, solution_field = fields
        config = (CONFIGS / "tasks" / config_name).read_text()
        assert f"dataset_dict_path: {dataset_path}" in config
        assert "load_dataset_dict: true" in config
        assert f"problem_field: {problem_field}" in config
        assert f"answer_field: {answer_field}" in config
        assert f"solution_field: {solution_field}" in config
        assert "normalize_dataset_fields: true" in config
        assert "use_dataset_answer: true" in config


@pytest.mark.parametrize(
    ("config_name", "dataset_path", "split"),
    (
        ("aime24_inplace_no_answer_prefix.jsonnet", "aime24", "test"),
        ("aime25_inplace_no_answer_prefix.jsonnet", "aime25", "test"),
        ("amc23_inplace_no_answer_prefix.jsonnet", "amc23", "test"),
        (
            "olympiadbench_hf_inplace_no_answer_prefix.jsonnet",
            "olympiadbench_hf",
            "train",
        ),
        ("collegeMath_inplace_no_answer_prefix.jsonnet", "collegeMath", "test"),
    ),
)
def test_local_eval_dataset_builds_task(config_name, dataset_path, split, monkeypatch):
    if not (DATA / dataset_path / "dataset_dict.json").exists():
        pytest.skip(f"data/{dataset_path} has not been downloaded")

    monkeypatch.chdir(ROOT)
    config_json = _jsonnet.evaluate_file(str(CONFIGS / "tasks" / config_name))
    task = Task.from_params(Params(json.loads(config_json)))
    dataset = task.get_datasets(split)

    assert len(dataset) > 0
    assert {"_treetune__idx", "problem", "answer", "query"} <= set(
        dataset.column_names
    )
    assert dataset[0]["query"] == dataset[0]["problem"]
    assert dataset[0]["answer"] not in (None, "", [])


@pytest.mark.parametrize(
    "config_name",
    (
        "deepseekR1Qwen_for_MATH_eval.jsonnet",
        "qwen1_5b_base_for_MATH_eval.jsonnet",
        "sft_deepseekmath_for_MATH_eval.jsonnet",
        "sft_rho1b_for_MATH_eval.jsonnet",
    ),
)
def test_math_eval_configs_compile_with_all_benchmarks(config_name):
    config = json.loads(_jsonnet.evaluate_file(str(CONFIGS / config_name)))
    inference_names = {
        pipeline["inference_name"] for pipeline in config["inference_pipelines"]
    }
    assert {
        "math_test",
        "aime24_test",
        "aime25_test",
        "amc23_test",
        "olympiadbench_test",
        "collegeMath_test",
    } <= inference_names


def test_smollm_eval_config_uses_one_model_and_small_gpu_limits():
    config = json.loads(
        _jsonnet.evaluate_file(
            str(CONFIGS / "polIter_smollm_135m_eval_MATH.jsonnet"),
            ext_vars={
                "APP_SEED": "42",
                "APP_DISABLE_FLASH_ATTENTION": "1",
            },
        )
    )
    model_name = "HuggingFaceTB/SmolLM2-135M"

    assert config["tokenizer"]["hf_model_name"] == model_name
    assert config["evaluation_vllm_server"]["max_num_seqs"] == 8
    assert config["evaluation_vllm_server"]["max_model_len"] == 1024

    for pipeline in config["inference_pipelines"]:
        strategy = pipeline["inference_strategy"]
        assert strategy["guidance_llm"]["model"] == model_name
        assert strategy["guidance_llm"]["tokenizer_name"] == model_name
        assert (
            strategy["node_expander"]["tokenizer"]["hf_model_name"]
            == model_name
        )
        assert strategy["node_expander"]["model_context_size"] == 1024
