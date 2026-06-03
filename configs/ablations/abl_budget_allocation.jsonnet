// Switch InGPO from legacy TV SHARE/PRUNE to simulation-lemma budget allocation.
// TV is used only for reward variance; branch-factor under-allocation is kept.
{
  ingpo+: {
    algorithm_mode: 'budget_allocation',
    budget_overhead_mode: 'flexible',
    tv_estimator: 'subnode',
    n_tv_estimates: 8,
    tv_subnode_max_tokens: 120,
    tv_second_phase_tokens: 60,
    budget_lambda: 0.02,
  },
}
