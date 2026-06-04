// Default InGPO hyperparameters from PLAN.md §3 / §4.
{
  ingpo: {
    // PLAN line 112: K=10, m=100
    K: 10,
    m: 100,

    // Lemma 2.4: eta = epsilon / R_max - exp(delta_avg). Setting eta_override
    // to null instructs the engine to derive it on the fly.
    epsilon: 0.02,
    r_max: 1.0,
    gamma: 0.5,
    alpha: 0.05,
    use_dkw: true,
    eta_override: null,

    // Triggers
    enable_share: true,
    enable_prune: true,
    share_target: 'nearest',  // 'nearest' | 'parent' | 'root'
    local_value_share: true,
    share_pair_budget_fraction: 0.25,  // roughly (W/2)^2 sibling pairs
    share_use_confidence: false,

    // Y generation
    y_temperature: 0.7,
    y_max_tokens: 512,
    y_field: 'answer',

    // Tail-fill concurrency to vLLM /completions
    score_concurrency: 64,

    // Algorithm mode. The default preserves the legacy InGPO SHARE/PRUNE
    // behavior; set to 'budget_allocation' to use simulation-lemma variance
    // allocation. In budget-allocation mode, TV is not used for SHARE/PRUNE.
    algorithm_mode: 'share_prune',
    tv_estimator: 'subnode',
    n_tv_estimates: 8,
    tv_subnode_max_tokens: 120,
    tv_second_phase_tokens: 60,
    tv_includes_half_factor: false,
    budget_lambda: 0.02,
    budget_overhead_mode: 'flexible',
    budget_queue_count: 2,
    budget_queue_timeout_seconds: 0.5,

    // Edge handling
    zero_advantage_when_pruned: false,
    emit_pruned_edges: true,

    // Logging: how many SHARE / PRUNE demo rows to dump per tree.
    // Set 0 to skip demo files entirely (per-depth rates stay on).
    demo_examples_per_tree: 2,

    // Where to write `demos.jsonl` + `demos.md` (offline-friendly). When
    // null they go under <APP_DIRECTORY>/<exp>/ingpo_demos/. Override with
    // an absolute path if you want them somewhere shared.
    demos_dir: null,

    // Off by default so an offline server with `wandb mode=offline` does
    // not try to upload tables. Flip to true if you do have wandb running.
    log_demos_to_wandb: false,
  },
}
