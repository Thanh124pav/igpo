// Cheaper budget-allocation TV estimate for overhead-sensitive runs.
(import 'abl_budget_allocation.jsonnet') + {
  ingpo+: { n_tv_estimates: 4 },
}
