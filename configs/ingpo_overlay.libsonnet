local default_ingpo = (import 'ingpo_defaults.libsonnet').ingpo;

// Reusable overlay that converts an SPO-tree config into an InGPO-tree
// config. Apply via:
//
//   (import 'polIter_<model>_<dataset>_spo_tree.jsonnet')
//   + (import 'ingpo_defaults.libsonnet')
//   + (import 'ingpo_overlay.libsonnet')
//
// The overlay reads `ingpo_cfg.*` (provided by ingpo_defaults.libsonnet)
// and mirrors the knobs onto the inference strategy / episode generator
// where SPO's main entrypoint will pass them to InGPO's __init__.
//
// Anything specific to a single experiment (eta, K, m, prune-only, ...)
// is supplied by ablation snippets that override `ingpo_cfg.*` BEFORE this
// overlay is applied. Because Jsonnet evaluates everything lazily, the
// final values come out the way the caller chained them.
{
  // Make this overlay safe even when a caller forgets to compose
  // `ingpo_defaults.libsonnet` before it.  The field is additive, so explicit
  // experiment/ablation overrides still win through Jsonnet object merging.
  ingpo+: default_ingpo,
  local ingpo_cfg = $.ingpo,

  episode_generator+: {
    type: 'ingpo_episode_generator',

    inference_strategy+: {
      type: 'ingpo',
      node_expander+: {
        store_logprobs: true,
      },
      ingpo_K: ingpo_cfg.K,
      ingpo_m: ingpo_cfg.m,
      ingpo_epsilon: ingpo_cfg.epsilon,
      ingpo_r_max: ingpo_cfg.r_max,
      ingpo_gamma: ingpo_cfg.gamma,
      ingpo_alpha: ingpo_cfg.alpha,
      ingpo_use_dkw: ingpo_cfg.use_dkw,
      ingpo_eta_override: ingpo_cfg.eta_override,
      ingpo_enable_share: ingpo_cfg.enable_share,
      ingpo_enable_prune: ingpo_cfg.enable_prune,
      ingpo_share_target: ingpo_cfg.share_target,
      ingpo_local_value_share: ingpo_cfg.local_value_share,
      ingpo_share_pair_budget_fraction: ingpo_cfg.share_pair_budget_fraction,
      ingpo_share_use_confidence: ingpo_cfg.share_use_confidence,
      ingpo_score_concurrency: ingpo_cfg.score_concurrency,
      ingpo_algorithm_mode: ingpo_cfg.algorithm_mode,
      ingpo_tv_estimator: ingpo_cfg.tv_estimator,
      ingpo_n_tv_estimates: ingpo_cfg.n_tv_estimates,
      ingpo_tv_subnode_max_tokens: ingpo_cfg.tv_subnode_max_tokens,
      ingpo_tv_second_phase_tokens: ingpo_cfg.tv_second_phase_tokens,
      ingpo_tv_includes_half_factor: ingpo_cfg.tv_includes_half_factor,
      ingpo_budget_lambda: ingpo_cfg.budget_lambda,
      ingpo_budget_overhead_mode: ingpo_cfg.budget_overhead_mode,
      ingpo_budget_queue_count: ingpo_cfg.budget_queue_count,
      ingpo_budget_queue_timeout_seconds: ingpo_cfg.budget_queue_timeout_seconds,
      ingpo_skip_near_leaf_expand: ingpo_cfg.skip_near_leaf_expand,
      ingpo_root_allocation: ingpo_cfg.root_allocation,
    },

    ingpo_zero_advantage_when_pruned: ingpo_cfg.zero_advantage_when_pruned,
    ingpo_emit_pruned_edges: ingpo_cfg.emit_pruned_edges,
    ingpo_demo_examples_per_tree: ingpo_cfg.demo_examples_per_tree,
    ingpo_demos_dir: ingpo_cfg.demos_dir,
    ingpo_log_demos_to_wandb: ingpo_cfg.log_demos_to_wandb,
  },
}
