(import 'math_inplace_no_answer_prefix.jsonnet') + {
  load_dataset_dict: false,
  dataset_dict_path: null,
  hf_dataset_args: ['Hothan/OlympiadBench', 'OE_TO_maths_en_COMP'],
  problem_field: 'question',
  answer_field: 'final_answer',
  solution_field: 'solution',
  normalize_dataset_fields: true,
  use_dataset_answer: true,
}
