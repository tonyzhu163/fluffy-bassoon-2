# Shared 1M-action LIBERO source pilot

The user selected **1,000,000 executed simulator actions per benchmark**:
LIBERO and RoboTwin separately, up to2M combined. The allowance includes both
SAC and fresh PPO source data, source validation, qualifications, settling,
retries and unresolved execution intents. It is not1M per method. The old
9,600-episode/160GPUh proposal is superseded and was never launched.

This plan owns LIBERO only; the user authorized informing the RoboTwin owner
in `Clarify DSLR online RL post-training`. No active RoboTwin contract is
modified from this checkout. Meta-World stays paused.

## Action reservations

| Source use | Total action reservation |
| --- | ---: |
| New SAC training: all40 source goals ×9,500 | 380,000 |
| New independent SAC local validation:8 fixed source goals ×12,000 | 96,000 |
| Completed source qualification: actual primitives+settling | 12,944 |
| Fresh PPO training/validation, including any separately trained reset control | 490,000 |
| Unassigned reserve, including earlier failed-qualification costs and retries beyond existing slots | 21,056 |
| Total | 1,000,000 |

SAC's ceiling is490k; the explicitly reserved SAC streams plus completed
qualification use at most488,944. The1,056 difference is not automatically
spent. Stream resumes retain their existing cap; extra attempts are never
silently assigned a fresh full reservation. Offline AD/DPT full/static models
reuse the same source interactions and do not each incur another corpus.

All40 training goals remain Spatial/Object/Goal/LIBERO10 tasks0–9. Local
validation uses tasks0 and1 in each of these four suites, selected before
outcomes, on initial-state IDs40–49. Training uses IDs0–39. This reduces
independent cold-teacher warmup costs; it is not new-goal validation. The
full LIBERO90, Plus and Pro target suites remain unchanged and contribute
zero source pretraining trajectories. No semantic goal audit/filter is used.

The source plan has512 **candidate** seeded cases per task/partition, each
with four episodes. These entries are not a demand or claim to collect512
cases. The collector starts another case only when its full worst-case
4×(400+10) actions fit. It publishes the final complete prefix when the
physical budget prevents another case. Report actual nominal state IDs,
state hashes, seed pairs, episodes, failures, query chunks and SAC updates.
Very short episodes can leave a stream below500 queried warmup chunks;
retain it as raw data rather than fabricate AD labels or extend its cap.

## Execution and data

SAC keeps the authors' batch256/UTD20 and warmup>500 queried chunks, with
fixed weights/stage within each episode and a persistent replay/optimizer
per task/partition. A source actor snapshot is a candidate, not an expert.

Fresh recurrent PPO stores its own behavior-policy rollouts. SAC trajectories
may support offline pretraining, but are not an ordinary PPO batch. Save public
camera/proprioceptive observations and next observations, task/instruction,
case/reset identity, actual latent noise, decoded executed motor actions,
raw and chunk rewards, duration/discount, success termination versus timeout,
trial boundaries, policy version, pre-tanh samples, behavior log probabilities,
values, recurrent cache/reset masks and full optimizer/sampler/RNG recovery.
Returns/GAE are derived using the saved configuration; they are not new data.

The one-task PPO driver is an allocated runtime building block. Shared all40
meta-policy orchestration must keep one policy/optimizer across source tasks;
independent fresh per-task PPO models do not satisfy that experiment.
The pilot's PPO batch must support repeated actual updates within its budget;
VLRLA's much larger rollout batch is not silently claimed as reproduced.

Source artifacts retain full episode/RPC traces and complete recovery at
episode boundaries (SAC) or trial boundaries (PPO), with sparse immutable
source checkpoint/prefix publication every16 cases plus final. Select one
latest committed paired prefix/history when assembling AD/DPT data.

## Pinned first stage

First SAC stream: `libero40-1m-sac-20261029-s00-train-r1`, source
`8ada0c0a59ea562716b251dc3cd7d71dc1ba3b18`, all40/fixed8 campaign hash
`9af3e471d21b535250c42899ee3d2bc81f25c4bdbed7a68fbe3052fb8242af2c`.
Schedule hash `a59bbbbb4b6050cd4a330d395347dd4edc6257b55beebd5c6a91e896355a0716`.
The full schedule stays on Baige beside the compact [campaign](campaign.json).

OneGPU/16CPU/96GiB/8GiB shared memory, at most40minutes, actual allocation
probe and focused Linux tests before model interaction. First verify capped
case completion, real postwarmup updates, finite losses, final paired exports,
complete recovery and physical-action accounting. More source streams can run
independently once this path passes; do not duplicate reserved task slots.

The1M action allowance does not establish data sufficiency or final performance.
Measure source improvement, informative feedback, robot query success and
equal-interaction controls. Report source/deployment cost separately. Source
training compute and target evaluation remain separately metered; a material
GPU-hour expansion requires the repository's concrete-budget discussion.

## Targeted-experiment allocation amendment, 2026-10-30

Before any full source-PPO launch, the full/reset caps are196k each rather
than200k. The490k PPO pool now retains19.5k prior qualification reservations,
392k full/reset training,64k frozen source validation,14k evaluation-only
escrow for the paired AD/DPT target diagnostic, and500 contingency. The
general21,056 reserve funds21,046 further target actions and10 verified
failed-qualification settling actions. All existing reservations and the
benchmark1M ceiling stay unchanged; target observations never enter source
pretraining. The joint35,046-action target cap covers28 primary method/case
groups including AD/DPT fixed-weight memory lesions. See
[the diagnostic protocol](../libero40-subset-target-20261030-r1/PROTOCOL.md).
The larger32GPU-hour request remains pending, and targeted results come first.
