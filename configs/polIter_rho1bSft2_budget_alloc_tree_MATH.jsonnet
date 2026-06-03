// Simulation-lemma budget-allocation tree on MATH with Rho-1.1B-SFT.
// Matched to the Rho SPO/GRPO MATH baselines.
(import 'polIter_rho1bSft2_spo_tree_MATH.jsonnet')
+ (import 'ingpo_defaults.libsonnet')
+ (import 'ablations/abl_budget_allocation.jsonnet')
+ (import 'ingpo_overlay.libsonnet')
