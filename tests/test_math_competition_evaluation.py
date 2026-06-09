from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_math_eval_configs_include_competition_benchmarks():
    eval_configs = (
        "deepseekR1Qwen_for_MATH_eval.jsonnet",
        "qwen1_5b_base_for_MATH_eval.jsonnet",
        "sft_deepseekmath_for_MATH_eval.jsonnet",
        "sft_rho1b_for_MATH_eval.jsonnet",
    )

    for config_name in eval_configs:
        config = (CONFIGS / config_name).read_text()
        assert "evaluation/math_competition_benchmarks.libsonnet" in config
        assert "] + competition_benchmark_pipelines" in config


def test_competition_benchmark_overlay_defines_all_requested_evals():
    overlay = (
        CONFIGS / "evaluation" / "math_competition_benchmarks.libsonnet"
    ).read_text()

    for inference_name in (
        "aime24_test",
        "aime25_test",
        "amc23_test",
        "olympiadbench_test",
    ):
        assert f"inference_name: '{inference_name}'" in overlay


def test_competition_tasks_use_normalized_math_dataset_fields():
    expected = {
        "aime24_inplace_no_answer_prefix.jsonnet": ("'problem'", "null", "'solution'"),
        "aime25_inplace_no_answer_prefix.jsonnet": ("'problem'", "'answer'", "null"),
        "amc23_inplace_no_answer_prefix.jsonnet": ("'question'", "'answer'", "null"),
        "olympiadbench_hf_inplace_no_answer_prefix.jsonnet": (
            "'question'",
            "'final_answer'",
            "'solution'",
        ),
    }

    for config_name, (problem_field, answer_field, solution_field) in expected.items():
        config = (CONFIGS / "tasks" / config_name).read_text()
        assert f"problem_field: {problem_field}" in config
        assert f"answer_field: {answer_field}" in config
        assert f"solution_field: {solution_field}" in config
        assert "normalize_dataset_fields: true" in config
        assert "use_dataset_answer: true" in config
