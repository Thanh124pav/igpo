// VinePPO on MATH with Rho-1.1B-SFT, mirroring the shipped GSM8K VinePPO setup.
(import 'polIter_rho1bSft2_spo_chain_MATH.jsonnet')
+ {
  episode_generator+: {
    type: 'vineppo_episode_generator',
  },
  trainer+: {
    params+: {
      use_prob_mask: false,
    },
  },
}
