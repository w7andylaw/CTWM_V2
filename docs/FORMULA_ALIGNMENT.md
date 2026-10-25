# CT-WM v13 formula alignment

Revision `ctm-identity-gated-retention-v13`. All75 equation labels remain;71 expressions and the entire
native task/reward section are unchanged from v12. Only activity retention,
environment carry, unrolled activity and constant-gate writing change. Rewards,
action-bank head, model/KL/tick/actor/critic losses and coefficients are identical.
Gate0 preserves activity, not all history/readout state. Gate1 recovers the
fully open old update. kappa remains the tested value; its old arbitrary-gate
one-step-write claim no longer holds. Current runtime/compiler evidence is
recorded separately; historical reports do not validate current source.

| Label | TeX line | Implementation |
|---|---:|---|
| `eq:benchmark_pomdp` | 68 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` |
| `eq:benchmark_transition` | 73 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` |
| `eq:benchmark_observation` | 79 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` |
| `eq:benchmark_history` | 84 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` |
| `eq:benchmark_return` | 89 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` |
| `eq:benchmark_discount` | 97 | `recall2imagine/embodied/envs/*; embodied/core/driver.py; native task specification` |
| `eq:world_model_state` | 135 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` |
| `eq:rssm_deterministic` | 146 | `recall2imagine/nets.py: RSSM.img_step (baseline comparison)` |
| `eq:ct_prior_context` | 151 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` |
| `eq:latent_feature` | 159 | `recall2imagine/agent.py: latent_input_config, WorldModel; behaviors.py` |
| `eq:control_feature` | 164 | `recall2imagine/agent.py: latent_input_config, WorldModel; behaviors.py` |
| `eq:reward_feature` | 169 | `recall2imagine/reward_nets.py: ActionConditionedMLP/ActionConditionedLinear; agent.py: WorldModel.reward` |
| `eq:reward_action_weights` | 179 | `recall2imagine/reward_nets.py: ActionConditionedMLP/ActionConditionedLinear; agent.py: WorldModel.reward` |
| `eq:reward_first_layer` | 188 | `recall2imagine/reward_nets.py: ActionConditionedMLP/ActionConditionedLinear; agent.py: WorldModel.reward` |
| `eq:transition_action_clip` | 204 | `recall2imagine/nrsm_nets.py: NRSM._action; agent.py: WorldModel reward-action alignment` |
| `eq:ct_tick_input` | 211 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` |
| `eq:ct_tick_initialization` | 211 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` |
| `eq:ct_synapse_glu` | 222 | `recall2imagine/nrsm_nets.py: _NavigationSynapseBlock, NRSM._transition` |
| `eq:ct_synapse_layernorm` | 227 | `recall2imagine/nrsm_nets.py: _NavigationSynapseBlock, NRSM._transition` |
| `eq:ct_synapse_tick` | 238 | `recall2imagine/nrsm_nets.py: _NavigationSynapseBlock, NRSM._transition` |
| `eq:ct_pre_history` | 238 | `recall2imagine/nrsm_nets.py: PrivateNLM.__call__, NRSM._transition` |
| `eq:ct_neuron_model` | 247 | `recall2imagine/nrsm_nets.py: PrivateNLM.__call__, NRSM._transition` |
| `eq:private_nlm` | 253 | `recall2imagine/nrsm_nets.py: PrivateNLM.__call__, NRSM._transition` |
| `eq:ct_write_gate` | 258 | `recall2imagine/nrsm_nets.py: PrivateNLM.__call__, NRSM._transition` |
| `eq:ct_time_constants` | 266 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` |
| `eq:ct_write_calibration` | 275 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` |
| `eq:ct_activity_retention` | 282 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` |
| `eq:ct_post_history` | 290 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` |
| `eq:ct_environment_decay` | 298 | `recall2imagine/nrsm_nets.py: NRSM._transition; independent analytic/numerical references` |
| `eq:ct_sync_selection` | 310 | `recall2imagine/nrsm_nets.py: NRSM._sync_indices, window_synchronization` |
| `eq:ct_sync_weights` | 318 | `recall2imagine/nrsm_nets.py: NRSM._sync_indices, window_synchronization` |
| `eq:ct_sync_state` | 326 | `recall2imagine/nrsm_nets.py: NRSM._sync_indices, window_synchronization` |
| `eq:ct_sync_context` | 336 | `recall2imagine/nrsm_nets.py: NRSM._transition, sync_projection` |
| `eq:ct_persistent_carry` | 336 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` |
| `eq:ct_dynamics_predictor` | 350 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` |
| `eq:ct_encoder` | 350 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` |
| `eq:categorical_unimix` | 357 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` |
| `eq:ct_latent_distributions` | 362 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` |
| `eq:categorical_st` | 368 | `recall2imagine/nrsm_nets.py: prior/posterior statistics; nets.py: distributions; jaxutils.py: OneHotDist` |
| `eq:zero_state_reset` | 376 | `recall2imagine/nrsm_nets.py: NRSM.initial, NRSM._transition, NRSM._obs_step` |
| `eq:tick_latent_cost` | 388 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` |
| `eq:tick_certainty` | 388 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` |
| `eq:error_tick_weight` | 396 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` |
| `eq:certainty_tick_weight` | 396 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` |
| `eq:nrsm_tick_loss` | 396 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` |
| `eq:credit_path` | 406 | `recall2imagine/nrsm_nets.py: tick_statistics; agent.py: WorldModel.loss` |
| `eq:prediction_factorization` | 418 | `recall2imagine/agent.py: WorldModel.loss, WorldModel.reward; reward_nets.ActionConditionedMLP` |
| `eq:symlog` | 428 | `recall2imagine/jaxutils.py: symlog, symexp, DiscDist; nets.py: Dist` |
| `eq:nrsm_observation_loss` | 436 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` |
| `eq:nrsm_image_loss` | 436 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` |
| `eq:twohot_target` | 446 | `recall2imagine/jaxutils.py: symlog, symexp, DiscDist; nets.py: Dist` |
| `eq:twohot_loss` | 452 | `recall2imagine/jaxutils.py: symlog, symexp, DiscDist; nets.py: Dist` |
| `eq:twohot_readout` | 452 | `recall2imagine/jaxutils.py: symlog, symexp, DiscDist; nets.py: Dist` |
| `eq:nrsm_reward_loss` | 459 | `recall2imagine/agent.py: WorldModel.loss, WorldModel.reward; reward_nets.ActionConditionedMLP` |
| `eq:nrsm_done_loss` | 467 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` |
| `eq:dynamics_kl_loss` | 474 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` |
| `eq:representation_kl_loss` | 474 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` |
| `eq:model_loss` | 481 | `recall2imagine/agent.py: WorldModel.loss; nets.py: dyn_loss, rep_loss; jaxutils.py: distributions` |
| `eq:imagined_actor` | 495 | `recall2imagine/agent.py: WorldModel.imagine; nrsm_nets.py: NRSM.img_step; behaviors.py` |
| `eq:imagined_context` | 495 | `recall2imagine/agent.py: WorldModel.imagine; nrsm_nets.py: NRSM.img_step; behaviors.py` |
| `eq:imagined_latent` | 495 | `recall2imagine/agent.py: WorldModel.imagine; nrsm_nets.py: NRSM.img_step; behaviors.py` |
| `eq:discrete_action_sample` | 507 | `recall2imagine/agent.py: ImagActorCritic; nets.py: Dist; jaxutils.py: distributions` |
| `eq:continuous_action_sample` | 513 | `recall2imagine/agent.py: ImagActorCritic; nets.py: Dist; jaxutils.py: distributions` |
| `eq:continuous_logprob` | 521 | `recall2imagine/agent.py: ImagActorCritic; nets.py: Dist; jaxutils.py: distributions` |
| `eq:imagined_reward` | 531 | `recall2imagine/agent.py: WorldModel.imagine/reward; behaviors.py: unchanged extrinsic reward slicing` |
| `eq:imagined_continuation` | 531 | `recall2imagine/agent.py: WorldModel.imagine; behaviors.py: inherited continuation weighting` |
| `eq:lambda_return` | 541 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` |
| `eq:return_normalization` | 547 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` |
| `eq:actor_loss` | 555 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` |
| `eq:value_loss` | 563 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` |
| `eq:slow_value_update` | 572 | `recall2imagine/agent.py: ImagActorCritic, VFunction; jaxutils.py: lambda_return, Moments, SlowUpdater` |
| `eq:ct_time_initialization` | 727 | `recall2imagine/nrsm_nets.py: NRSM time-scale/write calibration and _transition` |
| `eq:ct_unrolled_activity` | 737 | `recall2imagine/nrsm_nets.py: NRSM._transition; independent analytic/numerical references` |
| `eq:ct_constant_write` | 746 | `recall2imagine/nrsm_nets.py: NRSM._transition; independent analytic/numerical references` |
| `eq:ct_activity_bound` | 755 | `recall2imagine/nrsm_nets.py: NRSM._transition; independent analytic/numerical references` |
