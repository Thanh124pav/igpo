// Competition-math evaluations shared by all MATH training configurations.
// These datasets are loaded from Hugging Face on the first evaluation run.
function(base_pipeline) [
  base_pipeline + {
    task: (import '../tasks/aime24_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'aime24_test',
  },
  base_pipeline + {
    task: (import '../tasks/aime25_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'aime25_test',
  },
  base_pipeline + {
    task: (import '../tasks/amc23_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'amc23_test',
  },
  base_pipeline + {
    task: (import '../tasks/olympiadbench_hf_inplace_no_answer_prefix.jsonnet'),
    dataset_split: 'train',
    dataset_portion: 1,
    inference_name: 'olympiadbench_test',
  },
]
