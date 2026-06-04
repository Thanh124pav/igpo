{
  ingpo+: {
    algorithm_mode: 'budget_allocation',
    n_tv_estimates: 8,
    tv_subnode_max_tokens: 120,
    tv_second_phase_tokens: 60,
    budget_lambda: 0.02,
    budget_overhead_mode: 'flexible',
    demo_examples_per_tree: 0,
  },
  episode_generator+: {
    ingpo_log_reward_variance_nodes: true,
  },
}
