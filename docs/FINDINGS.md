# Verified findings

The strongest evidence so far is for native SFT competence and for some **source** policies improving under online RL. Paired frozen-student evaluations have not shown a consistent gain caused by retaining history. These are different measurements.

## RoboCasa365

| Measurement | Result | What it establishes |
| --- | --- | --- |
| [Native Human300 atomic SFT reproduction](../outputs/robocasa-sft-reproduction-20261109-r1/ATOMIC-RESULT.md) | 363 successes in 900 episodes (40.33%) across 18 atomic tasks, versus the published 39.6%. | The original JAX evaluator, checkpoint and horizons give a credible atomic reference. The [composite continuation](../outputs/robocasa-sft-reproduction-20261109-r1/CLOSEOUT.md) closed early and is not a composite benchmark score. |
| [Paired RLinf source endpoints](../outputs/robocasa-endpoint512-acceleration-20261104-r1/RESULT.md) | On 512 paired cases per task, OpenDrawer improved from 331 to 369 successes and CloseToasterOvenDoor from 168 to 212. Both passed the two-task Holm correction. | These fixed teachers improved over SFT on their own tasks. In a [separate three-task test](../outputs/robocasa-eight-hour-babysit-20261104-r1/SOURCE5-RESULT.md), Faucet passed (204 → 242/512); Counter and Coffee did not. This does not establish five effective sources. |
| [Frozen AD steering with real prior attempts](../outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/STAGE1-RESULT.md) | On held-out Fridge after five attempts, full-history KV and velocity students succeeded on 33 and 41 of 50 cases; clearing history gave 34 and 39. | Using the frozen expert improved some transfer comparisons with a direct action head. The full/reset comparisons show no consistent benefit from retaining history; one fit per interface. |
| [AD](../outputs/robocasa-eight-hour-babysit-20261104-r1/AD-ID50-RESULT.md) and [DPT](../outputs/robocasa-eight-hour-babysit-20261104-r1/DPT-OOD50-RESULT.md) direct-action baselines | Stronger performance on the three fitted tasks than on held-out Fridge and Microwave. | These students bypass the native action expert. Their performance cannot be attributed to KV steering or taken as proof of adaptation. |
| [Flora calibration](../outputs/robocasa-joint-h600-preparation-20261102-r4/RESULT.md) and [reward mismatch](../outputs/robocasa-ppo-reward-gap-investigation-20261102-r1/RESULT.md) | Native/task-specific AUROC ranged from 0.848 to 0.982 on five fixed 50-episode, 600-control banks. A separate OpenDrawer training audit found maximum grades on 58 of 183 simulator failures. | The rewarder ranks outcomes usefully on these panels, but a high grade is not native success. The matched 50-case residual-PPO comparisons did not establish superiority over simulator reward. |

A [controlled hidden-goal pilot](../outputs/robocasa-meta-feedback-20261109-r1/RESULT.md) executed three frozen π0.5 bowl prompts on each of eight scenes. Two of 12 support rollouts placed and released a fruit, both into the left bowl on one scene. None of the 12 fresh-query rollouts released into either bowl. These development cases provided zero successful query choices for testing reward memory or first-attempt exploration. A separate drawer manipulation check changed lateral miss at closure but yielded no handle acquisition or native success on the two query tapes examined.

For historical Flow-SDE teachers, the [critic parity audit](../outputs/robocasa-flow-sde-critic-parity-20261104-r1/RESULT.md) found that rollout and replay could use different denoising-step values even when likelihoods matched. Record the RLinf revision and critic definition when comparing these runs.

## RoboTwin

| Measurement | Result | What it establishes |
| --- | --- | --- |
| [Released Motus native phone baseline](../outputs/robotwin-motus-release-baseline-20261109-r1/RESULT.md) | 27/100 clean and 25/100 randomized successes on `place_phone_stand` under the documented reconstruction. | Below the reported 49%/53% for that task. This is one motor and controller protocol, not the 50-task aggregate. |
| [Corrected Motus source confirmation](../outputs/robotwin-restoration-20261106-r1/MOTUS-SOURCE-GRPO-CONFIRMATION01-READBACK.json) | SFT 7/64 vs. selected GRPO 10/64 on disjoint paired scenes; eight gains, five losses, exact two-sided p = .581. | The predeclared improvement gate failed. Working PPO/GRPO execution is not evidence of an improving source or transferable adaptation. |
| [Teacher-forced action replay](../outputs/robotwin-motus-replay-20261109-r1/RESULT.md) | On 20 clean demonstrations, mean shift-one arm error was 0.0154 rad versus 0.0789 for holding state. Wrist-camera swaps increased error. | The frozen motor predicts useful actions on expert observations. This offline measurement does not explain the closed-loop success gap. |

The [decoded-learner audit](../outputs/robotwin-restoration-20261106-r1/CPUDECODEDFIT01.json) and [four-method steering qualification](../outputs/steering-four-method-pipeline-20261103-r1/RESULT.md) test objectives, causal context and recovery. They are software qualifications, not held-out performance scores.

The [full pre-consolidation evidence ledger](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/24fcc5b016f6d6f15cd7bad09cec9bd723873338/docs/FINDINGS.md) preserves detailed benchmark results, negative findings and citations for retired experiments. Consult a run’s own source and evidence before comparing results across protocols.

On the frozen 100 clean phone-stand scenes, Sidney's original RoboTwin π0.5 SFT checkpoint scored 23/100 with the native 10 cm wrist-camera near plane and 75/100 with both wrists at 1 cm. In SAPIEN 3.0.1 the 10 cm setting clips the gripper fingertips visible in the training demonstrations. The paired intervention recovered 55 failures and lost three successes with equal seeds, instructions, starting state, head-camera pixels, noise and 400-target limit. All 813 published tensors matched the reference loader. The author reports 66/100 on a separately evaluated clean cohort. [Experiment, image comparison and paired readback](../outputs/robotwin-sidney-gap-20261109-r1/RESULT.md).

On the separate randomized phone-stand panel the same wrist-camera correction raised original Sidney SFT from 31/100 to 87/100. It recovered 58 failures and lost two successes. All 100 initial head images, joint states and prompts matched across camera conditions. [Randomized paired readback](../outputs/robotwin-sidney-gap-20261109-r1/WRIST-RANDOM-READBACK.json).
