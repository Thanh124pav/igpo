from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "treetune" / "inference_strategies" / "ingpo_inference_strategy.py"
DOC = ROOT / "docs" / "ingpo_pipeline_and_bugfixes.md"
DEFAULTS = ROOT / "configs" / "ingpo_defaults.libsonnet"


def test_share_prune_metadata_is_initialized_before_local_gate():
    text = STRATEGY.read_text()

    metadata_pos = text.index("# Attach InGPO metadata before local sibling gates run.")
    gate_pos = text.index("await _try_local_value_share_and_prune(node, children)")
    probe_pos = text.index("probe_tasks = []", gate_pos)

    assert metadata_pos < gate_pos < probe_pos
    assert "child.setdefault(\"ingpo_action\", Action.EXPAND.value)" in text
    assert "if child.get(\"ingpo_action\") != Action.EXPAND.value:" in text[probe_pos:]


def test_tv_half_factor_default_is_documented_and_enabled():
    assert "tv_includes_half_factor: true," in DEFAULTS.read_text()
    doc = DOC.read_text()
    assert "Total variation between two distributions is `0.5 * L1(p, q)`" in doc
    assert "SHARE/PRUNE gates were run before child metadata existed" in doc


def test_budget_allocation_caps_probe_budget_and_marks_completed_continuations():
    text = STRATEGY.read_text()

    assert "def _budget_tv_first_phase_tokens" in text
    assert "tokens = min(tokens, max(int(self.M), 1))" in text
    assert "first_phase_tokens=self._budget_tv_first_phase_tokens()" in text
    assert "max_tokens=self._budget_tv_first_phase_tokens()" in text

    completion_pos = text.index("async def _complete_candidate")
    continuation_pos = text.index("continuations = await _expand_with_budget", completion_pos)
    terminal_pos = text.index('if child.get("finish_reason") != "length":', continuation_pos)
    return_pos = text.index("return child", terminal_pos)

    assert continuation_pos < terminal_pos < return_pos
    assert 'child["leaf"] = True' in text[terminal_pos:return_pos]


def test_ingpo_overlay_has_defaults_fallback_and_store_logprobs():
    overlay = (ROOT / "configs" / "ingpo_overlay.libsonnet").read_text()

    assert "local default_ingpo = (import 'ingpo_defaults.libsonnet').ingpo;" in overlay
    assert "ingpo+: default_ingpo" in overlay
    assert "local ingpo_cfg = $.ingpo" in overlay
    assert "store_logprobs: true" in overlay
    assert "$.ingpo." not in overlay


def test_expander_accepts_store_logprobs_and_skips_zero_max_tokens():
    expansion = (
        ROOT
        / "treetune"
        / "inference_strategies"
        / "tree_inference"
        / "expansion.py"
    ).read_text()

    assert "store_logprobs: bool = False" in expansion
    assert "self.store_logprobs = bool(store_logprobs)" in expansion
    assert 'program_kwargs["logprobs"] = 1' in expansion
    assert "max_tokens is not None and int(max_tokens) <= 0" in expansion
    assert "if new_max_tokens <= 0:" in expansion
    assert "return []" in expansion
