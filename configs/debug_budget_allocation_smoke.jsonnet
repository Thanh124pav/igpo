// Tiny budget-allocation smoke overlay.
// Stack after a budget-allocation polIter config and an optional model override.
{
  num_iterations: 1,
  num_episodes_per_iteration: 2,
  evaluate_every_n_iterations: 1000,
  episodes_cloud_log_steps: 1,

  episode_generator+: {
    dataset_num_samples_per_iteration: 1,
    total_num_iterations: $.num_iterations,
    max_question_length: 256,
    vllm_gpu_memory_utilization: 0.2,
    vllm_min_available_gpu_memory_mb: 512,
    vllm_server+: {
      max_num_seqs: 16,
      swap_space: null,
      enable_prefix_caching: true,
    },
    inference_strategy+: {
      M: 32,
      max_depth: 2,
      max_concurrent_programs: 4,
      max_concurrent_generations: 4,
      branch_factor_strategy+: {
        // New tree_shape format: depth-0 has 2 nodes, depth-1 has 4 nodes.
        tree_shape: [2, 4],
      },
      node_expander+: {
        program_kwargs+: {
          max_tokens: 64,
          logprobs: 0,
        },
      },
    },
    value_estimation_inference_strategy+: {
      max_concurrent_programs: 4,
      max_concurrent_generations: 4,
    },
  },

  ingpo+: {
    algorithm_mode: 'budget_allocation',
    budget_overhead_mode: 'flexible',
    tv_estimator: 'subnode',
    n_tv_estimates: 4,
    tv_subnode_max_tokens: 16,
    tv_second_phase_tokens: 8,
    budget_lambda: 0.02,
    demo_examples_per_tree: 0,
    log_demos_to_wandb: false,
  },

  trainer+: {
    num_epochs_per_iteration: 1,
    general_training_args+: {
      target_train_batch_size: 2,
      per_device_train_batch_size: 1,
      per_device_eval_batch_size: 1,
      gradient_accumulation_steps: 1,
      save_steps: 1,
      checkpoint_keep_steps: 1,
    },
  },
}
