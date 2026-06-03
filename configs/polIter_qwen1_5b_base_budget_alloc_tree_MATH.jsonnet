// Simulation-lemma budget-allocation tree on MATH with Qwen2.5-1.5B-base.
// Matched to polIter_qwen1_5b_base_spo_tree_MATH / GRPO baselines.
(import 'polIter_qwen1_5b_base_spo_tree_MATH.jsonnet')
+ (import 'ingpo_defaults.libsonnet')
+ (import 'ablations/abl_budget_allocation.jsonnet')
+ (import 'ingpo_overlay.libsonnet')
