// Larger budget-allocation TV estimate for accuracy-oriented runs.
(import 'abl_budget_allocation.jsonnet') + {
  ingpo+: { n_tv_estimates: 16 },
}
