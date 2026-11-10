# Verified findings

The strongest evidence so far is for native SFT competence and for some **source** policies improving under online RL. Paired frozen-student evaluations have not shown a consistent gain caused by retaining history. These are different measurements.

## RoboCasa365

The Atomic-Seen baseline uses the native JAX π0.5 Human300 checkpoint at step **75,000**. Load `pi05_pretrain_human300/multitask_learning/75000` from `robocasa/robocasa365_checkpoints` at revision `c484448a`, including its bundled normalization. OpenPI predicts 50 actions and executes five before replanning, with ten denoising steps. [Pinned checkpoint](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/c484448aba1a9b60a04c9b0ca117241518ea69f3/pi05_pretrain_human300/multitask_learning/75000)

| Measurement | Result | What it establishes |
| --- | --- | --- |
| [Native Human300 atomic SFT reproduction](../outputs/robocasa-sft-reproduction-20261109-r1/ATOMIC-RESULT.md) | 363 successes in 900 episodes (40.33%) across 18 atomic tasks, versus the published 39.6%. | The original JAX evaluator, checkpoint and horizons give a credible atomic reference. The [composite continuation](../outputs/robocasa-sft-reproduction-20261109-r1/CLOSEOUT.md) closed early and is not a composite benchmark score. |
| [Paired RLinf source endpoints](../outputs/robocasa-endpoint512-acceleration-20261104-r1/RESULT.md) | On 512 paired cases per task, OpenDrawer improved from 331 to 369 successes and CloseToasterOvenDoor from 168 to 212. Both passed the two-task Holm correction. | These fixed teachers improved over SFT on their own tasks. In a [separate three-task test](../outputs/robocasa-eight-hour-babysit-20261104-r1/SOURCE5-RESULT.md), Faucet passed (204 → 242/512); Counter and Coffee did not. This does not establish five effective sources. |
| [Frozen AD steering with real prior attempts](../outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/STAGE1-RESULT.md) | On held-out Fridge after five attempts, full-history KV and velocity students succeeded on 33 and 41 of 50 cases; clearing history gave 34 and 39. | Using the frozen expert improved some transfer comparisons with a direct action head. The full/reset comparisons show no consistent benefit from retaining history; one fit per interface. |
| [AD](../outputs/robocasa-eight-hour-babysit-20261104-r1/AD-ID50-RESULT.md) and [DPT](../outputs/robocasa-eight-hour-babysit-20261104-r1/DPT-OOD50-RESULT.md) direct-action baselines | Stronger performance on the three fitted tasks than on held-out Fridge and Microwave. | These students bypass the native action expert. Their performance cannot be attributed to KV steering or taken as proof of adaptation. |
| [Flora calibration](../outputs/robocasa-joint-h600-preparation-20261102-r4/RESULT.md) and [reward mismatch](../outputs/robocasa-ppo-reward-gap-investigation-20261102-r1/RESULT.md) | Native/task-specific AUROC ranged from 0.848 to 0.982 on five fixed 50-episode, 600-control banks. A separate OpenDrawer training audit found maximum grades on 58 of 183 simulator failures. | The rewarder ranks outcomes usefully on these panels, but a high grade is not native success. The matched 50-case residual-PPO comparisons did not establish superiority over simulator reward. |

A [controlled hidden-goal pilot](../outputs/robocasa-meta-feedback-20261109-r1/RESULT.md) executed three frozen π0.5 bowl prompts on each of eight scenes. Two of 12 support rollouts placed and released a fruit, both into the left bowl on one scene. None of the 12 fresh-query rollouts released into either bowl. These development cases provided zero successful query choices for testing reward memory or first-attempt exploration. A separate drawer manipulation check changed lateral miss at closure but yielded no handle acquisition or native success on the two query tapes examined.

In the [compact-table follow-up](../outputs/robocasa-compact-bowls-20261109-r1/RESULT.md), frozen π0.5 grasped in 8/8 trials but all three released fruits went right, regardless of the requested side. A scripted post-grasp carry using simulator coordinates released to either requested side on 3/4 scenes (6/8 continuations). A fixed reward-conditioned rule on these same development traces obtained 3/4 query goals, versus 2/4 with either constant choice and 1/4 with inverted reward. There were only two scene pairs; the original motor did not qualify for held-out evaluation.

For historical Flow-SDE teachers, the [critic parity audit](../outputs/robocasa-flow-sde-critic-parity-20261104-r1/RESULT.md) found that rollout and replay could use different denoising-step values even when likelihoods matched. Record the RLinf revision and critic definition when comparing these runs.

## RoboTwin

### Sidney phone-stand baseline

All three policies start from `SidneyXie/pi05_robotwin` at revision `e49e2ab6`. The frozen evaluation uses the native per-target controller, a 1 cm near clipping plane on both wrist cameras, ten denoising steps, 50-action chunks and a 400-target episode limit. Within each domain, all three policies use the same 100 scenes, prompts and sampling seeds. [Pinned checkpoint](https://huggingface.co/SidneyXie/pi05_robotwin/tree/e49e2ab6c11f07511573b67261bd129e88d0a416) · [Matched evaluation](../outputs/robotwin-sidney-native-endpoints-20261109-r1/RESULT.md)

| Policy | Clean successes / 100 | Randomized successes / 100 |
| --- | ---: | ---: |
| Original SFT | 75 | 87 |
| PPO after 40 Adam updates | 77 | 88 |
| GRPO after 40 Adam updates | 78 | 90 |

**Training volume per algorithm** was two collections × 1,024 trajectories × 400 action slots = **819,200 nominal action steps**, or 16,384 chunks of 50. Each policy received 40 synchronized Adam updates. The training journal counted 842,240 slots including 23,040 setup/reset allowances. Actual delivered motor calls during training were unmeasured. [Training accounting](../outputs/robotwin-sidney-matched-baselines-20261106-r1/RESULT.md)

The subsequent frozen evaluation delivered **34,533 targets for PPO** and **33,768 for GRPO** across 200 episodes each, with zero further updates. The 1–3 point gains over SFT had two-sided exact paired p-values of at least 0.45. [PPO readback](../outputs/robotwin-sidney-native-endpoints-20261109-r1/PPO40-READBACK.json) · [GRPO readback](../outputs/robotwin-sidney-native-endpoints-20261109-r1/GRPO40-READBACK.json)

At the original 10 cm wrist clipping distance, SFT scored 23/100 clean and 31/100 randomized. Restoring the fingertips visible in training recovered 55 clean failures and lost three successes. On randomized scenes it recovered 58 and lost two. The reference loader matched all 813 released tensors. The uploader reports 66/100 clean on a separately evaluated cohort. [Camera experiment](../outputs/robotwin-sidney-gap-20261109-r1/RESULT.md)

### Motus and other RoboTwin results

| Measurement | Result | What it establishes |
| --- | --- | --- |
| [Released Motus native phone baseline](../outputs/robotwin-motus-release-baseline-20261109-r1/RESULT.md) | 27/100 clean and 25/100 randomized successes on `place_phone_stand` under the documented reconstruction. | Below the reported 49%/53% for that task. This is one motor and controller protocol, not the 50-task aggregate. |
| [Corrected Motus source confirmation](../outputs/robotwin-restoration-20261106-r1/MOTUS-SOURCE-GRPO-CONFIRMATION01-READBACK.json) | SFT 7/64 vs. selected GRPO 10/64 on disjoint paired scenes; eight gains, five losses, exact two-sided p = .581. | The predeclared improvement gate failed. Working PPO/GRPO execution is not evidence of an improving source or transferable adaptation. |
| [Teacher-forced action replay](../outputs/robotwin-motus-replay-20261109-r1/RESULT.md) | On 20 clean demonstrations, mean shift-one arm error was 0.0154 rad versus 0.0789 for holding state. Wrist-camera swaps increased error. | The frozen motor predicts useful actions on expert observations. This offline measurement does not explain the closed-loop success gap. |

The [decoded-learner audit](../outputs/robotwin-restoration-20261106-r1/CPUDECODEDFIT01.json) and [four-method steering qualification](../outputs/steering-four-method-pipeline-20261103-r1/RESULT.md) test objectives, causal context and recovery. They are software qualifications, not held-out performance scores.

The [full pre-consolidation evidence ledger](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/24fcc5b016f6d6f15cd7bad09cec9bd723873338/docs/FINDINGS.md) preserves detailed benchmark results, negative findings and citations for retired experiments. Consult a run’s own source and evidence before comparing results across protocols.
