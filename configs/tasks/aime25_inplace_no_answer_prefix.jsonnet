(import 'math_inplace_no_answer_prefix.jsonnet') + {
  load_dataset_dict: false,
  dataset_dict_path: null,
  hf_dataset_args: ['math-ai/aime25'],
  problem_field: 'problem',
  answer_field: 'answer',
  solution_field: null,
  normalize_dataset_fields: true,
  use_dataset_answer: true,
}
