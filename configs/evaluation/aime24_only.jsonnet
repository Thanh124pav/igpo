// Keep only the AIME 2024 evaluation pipeline from a MATH eval config.
{
  local aime24_pipelines = [
    p
    for p in super.inference_pipelines
    if std.objectHas(p, 'inference_name') && p.inference_name == 'aime24_test'
  ],

  assert std.length(aime24_pipelines) > 0 :
    'AIME24 pipeline not found. Use a MATH eval config that imports evaluation/math_benchmarks.libsonnet.',

  inference_pipelines: aime24_pipelines,
}
