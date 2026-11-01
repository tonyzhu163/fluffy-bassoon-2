# Native VLA PPO readiness: existing RLinf path

Static preparation only,2026-10-31. AD100/500 implementation is on hold; no code was edited for that profile and no allocation or model/test was run. Latest fetched origin/main was reviewed atacca706d5e75d1da099ff7ce6fddd727dace0fba. Existing active artifacts/worktrees remain unchanged.

## What can be run with existing code

Baige already has RLinf`c605b9d55ed93a180119b099e6f836abff293b08` and RoboTwin`dca9ec682688821c42944eacd7f5dd9bfb15396f` under`/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/`. The real upstream entry is`RLinf/examples/embodiment/train_embodied_agent.py`; the official pi0.5 PPO configuration is`examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml`. `run_embodiment.sh` is the standard launcher, but direct Hydra invocation makes all overrides explicit.

This entry uses the existing EmbodiedRunner and FSDP actor/rollout/environment worker groups. `algorithm.loss_type=actor_critic` selects PPO rather than the GRPO actor-only route. `openpi.task=rl,train_expert_only=True` freezes the VLM after loading weights (`openpi/__init__.py:286–296`); the native action expert and value head learn. FlowSDE samples one denoising integration step (`tasks/rl.py:235–258`), noise coefficient.3, five integration steps. This is a native-VLA PPO route, unlike our frozen-motor/GTrXL initial-noise policies.

Public released initializer already present:
`/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-sft-adjust-bottle-fa8df6ed`, HF`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed103db0f5549c122f3a17c00ba6426c98`. Use its own`physical-intelligence/robotwin/norm_stats.json` (SHAbff5fcadaba626c28c1c2350d3baa7f1a04befc9f8b73f3e558a09c1b9c6ee52), not the Motus or LIBERO normalization. Existing direct-VLA runtime is`/mowenjie/fluffy-bassoon-2/envs/robotwin-baseline-smoke-20261028-r1/bin/python`; qualified prior receipt records Torch2.11.0+cu128/SAPIEN3.0.1/MPLib.2.1/RLinfOpenPI.1.1.

The prior native-action GRPO learning gate job-agat344mm90m reached a nonzero gradient and step1checkpoint with this initializer:64training trajectories,53successes,42.212s actor training/536.866s native iteration. That was actor-only GRPO with group filtering, one epoch and reduced collection. Its82.8%is a pre-update training-batch rate, not a baseline128-case evaluation, post-training gain, PPO result or VLRLA reproduction. The original checkpoint plus all old runs stay intact.

## Paper-recipe mapping and actual gaps

The saved VLRLA protocol specifies25PPO iterations;256train environments×4rollouts×400primitive-step slots;50action chunks→8decisions/environment/round→8192nominal chunk slots/iteration; five optimization epochs/global minibatch2048/microbatch32; eightGPUs/FSDP/offloading. GAE.99/.95, normalized advantages, ratio±.2/dualclip3, clippedHuber critic10/valueclip.2/weight1, no entropy/referenceKL, AdamW actor5e-6/value1e-4/betas(.9,.95)/decay.01/norm1. Evaluation is128fixed-seed environments before training and every5iterations. Published valid post-terminal counts are unknown.

The installed upstream pi0.5 YAML matches most of those optimizer/model settings but defaults to **H200,maxepochs1000,evaluation every10**. Reconstructing the paper recipe requires explicit H400for train/eval wrapper+rollout+taskcutoff,25iterations and evaluationevery5. Keep256/4/2048/fiveepochs and50action chunk; reducing these to our current sixK4 source batches would cease to reproduce the recorded rollout recipe. Smaller execution microbatches/accumulation may be used only while preserving global loss weighting and schedules.

The dated2026-10-29source audit found no identified official VLRLA code/checkpoint/custom history dataset. That is bounded historical evidence, not a current assertion that none exists; root is refreshing primary-paper/release evidence. Exact paper initializer/SFTexposure, normalization, source-code/task commits and several student protocols remain unreported. Without a released author artifact or verified match, this path is a **paper-recipe reconstruction on the public RLinf initializer**, not an exact VLRLA result reproduction. A task-specific adjust_bottle initializer also cannot establish the paper's12-task mean. Single-task and whole-table claims must stay separate.

One code-side prerequisite is concrete: the installed `EmbodiedRunner.run()` starts collection/training directly, then evaluates in `_maybe_eval_and_checkpoint` after updates. It does **not** branch on `runner.only_eval` or automatically evaluate step0; that field alone must not be used to claim a baseline. The existing `runner.evaluate()`/`update_rollout_weights()` methods can supply the necessary pre-training128case readout within the same native runner; root must bind this small execution choice before launch. No new RL algorithm is needed.

## Prepared command shape (not executed or funded)

Use an actual allocation and reviewed clean code/runtime, exact task assets and the released initializer. The source config's placement is eight GPUs; availability/capacity/time must be bound by the execution owner. No new scheduler request is made here.

```bash
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export PYTHONPATH="$REPO_PATH:$ROBOTWIN_PATH"
/mowenjie/fluffy-bassoon-2/envs/robotwin-baseline-smoke-20261028-r1/bin/python  "$EMBODIED_PATH/train_embodied_agent.py" --config-path "$EMBODIED_PATH/config"  --config-name robotwin_adjust_bottle_ppo_openpi_pi05  runner.max_epochs=25 runner.max_steps=25 runner.val_check_interval=5 runner.save_interval=5  actor.model.model_path=/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-sft-adjust-bottle-fa8df6ed  ++actor.model.openpi_data.norm_stats_path=/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-sft-adjust-bottle-fa8df6ed/physical-intelligence/robotwin/norm_stats.json  env.train.assets_path="$ROBOTWIN_PATH" env.eval.assets_path="$ROBOTWIN_PATH"  env.train.max_episode_steps=400 env.train.max_steps_per_rollout_epoch=400 env.train.task_config.step_lim=400  env.eval.max_episode_steps=400 env.eval.max_steps_per_rollout_epoch=400 env.eval.task_config.step_lim=400  runner.logger.log_path=FRESH_REVIEWED_OUTPUT runner.logger.experiment_name=FRESH_NATIVE_PPO_ID
```

This command shape does not solve baseline-step0or source initializer ambiguity and is not a ready authority to spend compute. Keep actual pre/post evaluation cases/sampling convention paired, save native action-expert/value/optimizer/RNG recoveries and valid rollout masks, and report actual GPUtime/interactions. The old GRPO gate's disabledtorchcompile/no-shard options were runtime choices, not paper settings, and are not silently copied above. Existing renderer/process cleanup fixes may be reused only through the qualified native integration; no installed dependencies or active runs were modified here.

`code-inventory.json` pins existing entry/config/model-code/runner/seed-list bytes and the prior real direct-VLA-training receipt. No additional source collection, student fitting, tests, inference or allocation occurred in this readiness work.
