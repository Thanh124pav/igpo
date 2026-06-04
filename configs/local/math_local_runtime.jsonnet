// Local smoke-run defaults. Keep this last, after model/task/algorithm overlays.
{
  evaluate_every_n_iterations: 1,
  num_iterations: 2,
  num_episodes_per_iteration: 27,
  episodes_cloud_log_steps: 1,

  episode_generator+: {
    dataset_num_samples_per_iteration: 1,
    total_num_iterations: $.num_iterations,
    vllm_gpu_memory_utilization: 0.35,
    vllm_min_available_gpu_memory_mb: 1024,
    wait_until_memory_release: true,
    inference_strategy+: {
      max_concurrent_programs: 16,
      max_concurrent_generations: 16,
    },
    value_estimation_inference_strategy+: {
      max_concurrent_programs: 16,
      max_concurrent_generations: 16,
    },
    vllm_server+: {
      swap_space: null,
      max_num_seqs: 64,
      enable_prefix_caching: true,
    },
  },

  trainer+: {
    num_epochs_per_iteration: 1,
    general_training_args+: {
      target_train_batch_size: 27,
      per_device_train_batch_size: 1,
      per_device_eval_batch_size: 1,
      gradient_accumulation_steps: null,
      save_steps: 1,
      checkpoint_keep_steps: 1,
      logging_steps: 1,
    },
  },
}
