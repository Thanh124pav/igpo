import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"


def _text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def test_refactor_layout_removes_legacy_spo_tree():
    assert not any((ROOT / "spo").rglob("*.*"))


def test_deepseek_r1_qwen_files_are_not_named_qwen1b():
    qwen1b_paths = [
        path
        for root in (CONFIGS, SCRIPTS)
        for path in _text_files(root)
        if "qwen1b" in path.as_posix()
    ]
    assert qwen1b_paths == []

    deepseek_model_paths = [
        path.relative_to(ROOT).as_posix()
        for root in (CONFIGS, SCRIPTS)
        for path in _text_files(root)
        if "DeepSeek-R1-Distill-Qwen-1.5B" in path.read_text()
    ]
    assert deepseek_model_paths
    assert all("deepseekR1Qwen" in path for path in deepseek_model_paths)


def test_config_imports_resolve_inside_unified_configs_tree():
    import_re = re.compile(r"import\s+'([^']+)'")
    missing = []
    legacy_spo_imports = []
    for path in _text_files(CONFIGS):
        text = "\n".join(line.split("//", 1)[0] for line in path.read_text().splitlines())
        for imported in import_re.findall(text):
            if "spo/configs" in imported:
                legacy_spo_imports.append((path.relative_to(ROOT).as_posix(), imported))
                continue
            candidates = [
                (path.parent / imported).resolve(),
                (CONFIGS / imported).resolve(),
                (ROOT / imported).resolve(),
            ]
            if not any(candidate.is_file() for candidate in candidates):
                missing.append((path.relative_to(ROOT).as_posix(), imported))

    assert legacy_spo_imports == []
    assert missing == []


def test_ingpo_defaults_use_budget_allocation_not_share_prune():
    defaults = (CONFIGS / "ingpo_defaults.libsonnet").read_text()
    overlay = (CONFIGS / "ingpo_overlay.libsonnet").read_text()

    assert "algorithm_mode: 'budget_allocation'" in defaults
    assert "enable_share: false" in defaults
    assert "enable_prune: false" in defaults
    assert "ingpo_algorithm_mode: ingpo_cfg.algorithm_mode" in overlay
    assert "ingpo_tv_estimator: ingpo_cfg.tv_estimator" in overlay
    assert "store_logprobs: true" in overlay
