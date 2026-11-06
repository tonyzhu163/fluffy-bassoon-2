# RLinf RoboTwin recipe audit

Captured on 2026-11-06 while the frozen initial-policy diagnostic runs. The qualified runtime remains at RLinf `c605b9d55ed93a180119b099e6f836abff293b08`; the read-only current-main snapshot is `e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d` (committed 06:16:01 UTC). Public YAML snapshots and file hashes are retained in [upstream/SOURCES.json](upstream/SOURCES.json). No active runtime, model, source worktree or contract was upgraded.

## Which recipes are released

RLinf's [RoboTwin guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html) documents pi0/pi0.5 PPO on `adjust_bottle`, OpenVLA-OFT PPO/GRPO on `place_empty_cup`, and Lingbot-VLA GRPO on `click_bell`. Its pi0.5 initializer is task-specific SFT. The guide does not list a Sidney pi0.5 phone-stand GRPO preset.

The pi0.5 PPO, OpenVLA-OFT GRPO and pi0.5 standalone-evaluation task YAMLs, plus the pi0.5/OpenVLA model defaults, are byte-identical between the pinned and captured current commits. The inherited `robotwin_adjust_bottle` environment default has changed: current main explicitly sets `camera_near: 0.01`; the pin retains RoboTwin's native 0.1 m. These are distinct statements: unchanged task YAMLs do not imply unchanged environment implementation or rendering. The current `place_empty_cup` environment default was not retrieved; its pinned default is retained.

## Compare actual completed runs

The actual sources of truth for our runs are [PPO resolved config](../robotwin-sidney-matched-baselines-20261106-r1/ppo02/resolved-native-config.json), [GRPO resolved config](../robotwin-sidney-matched-baselines-20261106-r1/grpo01/resolved-native-config.json), their pinned code, and [completion evidence](../robotwin-sidney-matched-baselines-20261106-r1/RESULT.md).

| Setting | Released pi0.5 PPO | Released OpenVLA-OFT GRPO | Our completed Sidney pair |
| --- | --- | --- | --- |
| Initializer / task | Task-specific pi0.5 SFT / adjust bottle | Task-specific OpenVLA-OFT SFT / empty cup | Untouched broad Sidney SFT / phone stand |
| Embodiment / views | ALOHA-AgileX, head and two wrists, clean | Inherited paired Piper, head camera, randomized | ALOHA-AgileX, three Large_D435 views, clean |
| Action chunk / horizon | 50 / 200 | 25 / 200 | 50 / 400; 400 is also the pinned phone task limit |
| Train sampler | Five-step Flow-SDE, noise 0.3 | OpenVLA sampling temperature 1.6 | Five-step Flow-SDE, noise 0.3 |
| Advantage / critic | GAE 0.99/0.95; critic | Group-relative; no critic | Native GAE + critic for PPO; native group-relative actor for GRPO |
| Scene-group size | 1 | 8 | 8 for both arms; actual scene/instruction identities and eight replicas qualified |
| Relative binary success scale | 1 | 5 | 1, first-success relative reward |
| Reward/logprob level | Chunk / chunk | Action / token | Chunk / chunk |
| Reward filtering | Off | On, bounds 0.5 and 4.5 | Off |
| Train auto-reset / ignore terminations | Off / off | Off / off | Off / off |
| Ratio clips, low/high | 0.2 / 0.2 | 0.2 / 0.28 | 0.2 / 0.2 |
| Actor LR; Adam beta2 / epsilon | 5e-6; 0.95 / 1e-8 | 1e-4; 0.999 / 1e-5 | 5e-6; 0.95 / 1e-8 |
| Global optimization batch | 2,048 | 1,024 | 2,048 |
| Update epochs per collection | 5 | Not declared in this task YAML | 5 |
| Environments x collection rounds | 256 x 4 | 128 x 8 | 256 x 4 |
| Reference KL / entropy weights | 0 / 0 | 0 / 0 | 0 / 0 |
| Validation cadence | Every 10 outer iterations | Every 20 outer iterations | Disabled in execution pilot; no step-40 evaluation yet |
| Executed training duration | YAML cap 1,000 epochs; not a claimed convergence requirement | YAML cap 1,000 epochs; not a claimed convergence requirement | Two outer iterations, 40 Adam steps per rank |

The numerical native GRPO learner was qualified independently. Both arms have nonzero action-expert gradients, changed trainable weights, unchanged frozen weights and independently restored full owned optimizer state. These checks establish that updates execute correctly; they do not establish useful learning.

Our GRPO arm deliberately keeps the PPO-derived Sidney motor and common optimization settings while changing advantage, loss and critic presence. It is an RLinf GRPO adaptation, not a literal released OpenVLA-OFT or Lingbot recipe. OpenVLA normalization, token-level losses and model-specific sampling settings should not be copied directly to pi0.5.

The completed pair used the released fast observation path. Its accounting records requested/conservative action slots; it does not provide measured delivered controls within a terminating chunk. The full-observation diagnostic retains native `executed_action_count` masks, while the fast diagnostic preserves the previous behavior and reports that count as unknown. Chunk-boundary termination masking and a measured partial-chunk action count must not be conflated. A student corpus still requires separate physical-duration and action/mask qualification.

Inspection of the exact pinned simulator and LeRobot sources found a larger deployment difference. LeRobot calls native `take_action` once per target. RLinf calls `gen_sparse_reward_data` on all 50 targets, compresses/time-parameterizes the complete path and downsamples it to at most 500 physics points per arm. LeRobot's default prompt for this task is `place phone stand`, while RLinf uses a generated episode instruction. The first diagnostic keeps that RLinf controller/prompt fixed even for the original LeRobot model, so it isolates policy inference but does not reproduce LeRobot's complete deployment. The follow-up [controller/prompt contract](../robotwin-sidney-deployment-diagnostic-20261106-r1/PLAN.md) isolates both factors and measures actual physics steps before further training decisions.

The current Lingbot GRPO preset further illustrates that there is no single universal GRPO hyperparameter set: it declares two update epochs, ten flow steps, noise 0.7, actor LR 5e-5, KL/entropy coefficients 0.05, global batch 1,024 and no reward filtering. Those values belong to another motor/task.

## Implications for the next comparison

1. Finish the initial-policy diagnostic before selecting another training change. It separates ten versus five inference steps, ODE versus training SDE, and full versus fast observations with fixed cases, horizon, checkpoint and batch size. Batch-one success is not a direct reproduction of the earlier batch-64 training collector.
2. Compare saved PPO/GRPO endpoints with the initial policy on the same evaluation cases and inference protocol before ranking them. Existing 176/1,024, 224/1,024 and 184/1,024 counts are training collections at steps 0 and 20 on successive cohorts, not evaluations of the final step-40 policies.
3. Keep the measured training execution batch: four GPUs x microbatch 64 x accumulation 8 = global 2,048, five update epochs, qualified mixed BF16 with FP32 vision/norms. Changing execution batch must not silently change effective global batch, loss weights or optimizer schedule.
4. Sparse reward contrast is a measured concern: 61/128 first-collection groups were constant in both arms; 51/128 remained constant in the second GRPO collection. Filtering changes the training distribution and admitted sample count, so it requires an explicit comparison rather than an undocumented switch.
5. Longer training remains a hypothesis. VLRLA's detailed 25-iteration schedule describes its PPO history collector; its exact GRPO training budget is not fully specified. Neither the paper nor the released YAML caps establish that a particular number of extra GRPO steps will fix this run. The existing cumulative 50 GPU-hour cap remains binding.

Sources: the commit-pinned public snapshots linked above; [RLinf RoboTwin guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html); [VLRLA Appendix A.2.1 and Table S2](https://arxiv.org/html/2610.19517v1#A1.SS2.SSS1); [local VLRLA protocol review](../../docs/literature/vlrla-protocol.md). These source settings are evidence for differences, not demonstrated causes of a performance gap.
