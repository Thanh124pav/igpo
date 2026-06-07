// RLOO on MATH with the local DeepSeek-R1-Distill-Qwen-7B checkpoint.
// Reuse the GRPO rollout/trainer setup and switch only the group baseline.
(import 'polIter_deepseekR1Qwen7B_grpo_MATH.jsonnet')
+ {
  episode_generator+: {
    adv_method: 'rloo',
  },
}
