{
  episode_generator+: {
    task+: {
      dataset_dict_path: 'data/math-local-100',
    },
  },
  inference_pipelines: [
    p {
      task+: {
        dataset_dict_path: 'data/math-local-100',
      },
      dataset_portion: 1,
    }
    for p in super.inference_pipelines
  ],
}
