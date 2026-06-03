import importlib.util
import logging
import sys
import types
from pathlib import Path


def _load_branch_factor_module():
    class FakeRegistrable:
        @classmethod
        def register(cls, *args, **kwargs):
            def decorator(subclass):
                return subclass
            return decorator

    fake_common = types.ModuleType("treetune.common")
    fake_common.Registrable = FakeRegistrable
    fake_logging = types.ModuleType("treetune.logging_utils")
    fake_logging.get_logger = logging.getLogger
    old_common = sys.modules.get("treetune.common")
    old_logging = sys.modules.get("treetune.logging_utils")
    sys.modules["treetune.common"] = fake_common
    sys.modules["treetune.logging_utils"] = fake_logging
    try:
        path = Path("treetune/inference_strategies/tree_inference/branch_factor_strategy.py")
        spec = importlib.util.spec_from_file_location("branch_factor_strategy_under_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if old_common is not None:
            sys.modules["treetune.common"] = old_common
        else:
            sys.modules.pop("treetune.common", None)
        if old_logging is not None:
            sys.modules["treetune.logging_utils"] = old_logging
        else:
            sys.modules.pop("treetune.logging_utils", None)


ListBranchFactor = _load_branch_factor_module().ListBranchFactor


def test_tree_shape_converts_cumulative_counts_to_branch_factors():
    strategy = ListBranchFactor(tree_shape=[6, 36, 216])
    assert strategy({"depth": 0}) == 6
    assert strategy({"depth": 1}) == 6
    assert strategy({"depth": 2}) == 6


def test_legacy_branch_factor_config_still_works(caplog):
    with caplog.at_level(logging.WARNING):
        strategy = ListBranchFactor(
            branch_factors=[
                {"depth": 0, "branch_factor": 6},
                {"depth": 1, "branch_factor": 5},
            ]
        )
    assert strategy({"depth": 0}) == 6
    assert strategy({"depth": 1}) == 5
    assert "legacy branch_factors" in caplog.text
