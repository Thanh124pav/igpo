{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 3,
      branch_factor_strategy+: {
        // New tree-shape format: cumulative node counts at each depth.
        // This is equivalent to the legacy 6-6-6 branch-factor config.
        tree_shape: [6, 36, 216],
      },
    },
  },
}
