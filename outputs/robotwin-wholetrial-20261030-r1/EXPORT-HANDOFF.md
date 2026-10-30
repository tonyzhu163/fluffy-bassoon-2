# RoboTwin AD/DPT adapter handoff

2026-10-30. Active collector pin: `db0184089fa88a5f96c153f9b80029bb91b77def`.
This handoff describes the native format; it does not qualify a supervised
student, a mean-action teacher, or a LIBERO adapter on RoboTwin.

## Motor, features and policy inputs

- Frozen Motus pi0.5, native H32 × 32 noise coordinates, physical noise bound
  1.5. The motor executes 14 joint/gripper targets per primitive command.
- Public features are 2,062 float32 values: masked mean of the frozen final
  2,048-dimensional VLM prefix plus 14 normalized proprioceptive values.
  Reuse the checkpoint's frozen normalization and ALOHA transforms; do not fit
  new statistics on target data.
- `meta_rl.vla_noise_ppo.policy_inputs` produces 3,090 values: public features,
  previous **physical** steering noise (1,024), previous reward, boundary flag,
  clamped episode-index clock and primitive-step/current-task-horizon clock.
- GTrXL is width 128, two layers, four heads, memory 300 and gate bias 2.
  Persistent memory crosses episodes within a K4 trial. The trained reset
  control clears its memory and previous noise/reward at each episode boundary.
- Source limits are handover 600, scan 500 and the other four tasks 400.
  Use each recorded episode limit. New data save it explicitly; older data
  retain the separate 400-step contract.

## Checkpoints and behavior identity

`meta_rl.vla_noise_ppo.RecurrentNoisePPO` loads the saved PPOConfig, observation
width, noise shape and full learner state. Full payloads use
`kind=gtrxl_noise_ppo`, schema version 1, and include policy, optimizer, sampler,
global RNG and counters. A coordinator nests these under `states[method]`.
Individual/stage recoveries nest them under `learner`. Reconstruct teacher
memory from actual history using its own reset rules; never reuse a behavior
cache from another checkpoint. Deterministic physical labels would be
`1.5 * tanh(mean)`, reshaped to H32 × 32, and require separate source competence
qualification under that convention.

`meta_rl.vla_noise_sac.NoiseSAC` is reactive, with observation width 2,062,
action width 1,024 and three 128-wide LayerNorm/Tanh hidden layers. Full recovery
contains actor, twin Q/target Q, temperature, all three optimizers, replay,
interaction/training RNG streams and counters. Instantiate from its saved
contract/config/seed before loading. Source and existing source validation use
stochastic actor draws; competent deterministic-mean labels are not established.

`BehaviorPolicyArchive` in `scripts/diagnostics/robotwin_noise_compare.py`
writes content-addressed acting weights. PPO files have `kind=ppo_actor_value`;
SAC files have `kind=sac_actor`, `state_dict` and `parameter_sha256`. Descriptors
include file SHA256 and `policy_version=METHOD:update:N:PARAMETER_HASH`.
Warmup is explicitly `bounded_uniform_warmup` with no actor checkpoint: those
actions must not be attributed to the untrained Gaussian actor. The first
non-warmup actor action may still precede the first SGD update.

## Corpus and eligibility

The schema is `robotwin_public_decisions_v1`, implemented in
`src/fluffy_bassoon/rl/robotwin_corpus.py`. Each episode has reset.json and a
manifest; each chunk has JSON plus decision/native NPZ references. Content-
addressed frames retain original RGB, proprioception, language and frozen
features. Decision arrays include actual sampled noise and issued motor
commands. Metadata also saves PPO pre-squash samples, old log probability,
value, full public policy input, reset mask and feedback.

Statuses progress from intent to native_complete to complete. Require committed
manifests, matching file/array hashes and no unresolved decision. Preserve
native rewards, terminations/truncations, executed duration and valid-action
mask; success can execute only part of the issued motor chunk. The full steering
noise remains a 1,024-dimensional decision target even for a partial motor chunk.

Use only `context.phase=source`. Exclude preflight, source validation, all target
and native-reference records. Group four completed episodes by source trial ID;
then group same-task trials chronologically across actual policy versions for
AD. Preserve actual historical actions and feedback bitwise for paired AD/DPT
queries. Update IDs, partition/task audit IDs and checkpoint IDs are provenance,
not extra observations. Reject or explicitly quarantine incomplete/budget-cut
trials while retaining their physical cost in the ledger.

Old immutable stage bank: robotwin-staged-20261030-r2/experiment/stages/32000.
It uses the old 400-step contract. New preferred native snapshots will appear
at each worker's experiment/stages/iteration-10, -25, -50 and -100. Those are
independent immutable source snapshots; do not read a moving recovery as if it
were a fixed teacher or source prefix.

## Implementation status

Current RoboTwin source SAC and direct persistent/reset PPO are qualified and
running. The exporter's native adapter at
`9c27e027ffed1f66a5d56c79e142e7a55faba717` passed 34 allocated Linux tests and
a real SAC saved-data export: 30 complete K4 trials, 120 episodes and 47,602
source actions. Allocation-only runtime identity, acting tensor digests and
frozen source/task counter checks are verified. See
[results](../robotwin-native-export-20261030-r2/RESULTS.md). No independent
supervised static/reset student or bound-1.5 deployment adapter is qualified
yet. LIBERO's 4,104-feature,
tied-32D/H10, bound-1 exporter/student cannot be force-cast into this format.
Metadata conversion and teacher/student inference must run on allocated Linux;
development SSH is only for setup, transfers and read-only artifact inspection.

An early immutable prefix is available now, without waiting for iteration 10:
`/mowenjie/fluffy-bassoon-2/exports/robotwin-native-prefix-20261030-r1/manifest.json`.
Its SHA256 is `05680fb416e6b8031f04b9832e903906ea2c4cb3ee091499bf4c65eb54d4c37b`.
SAC has five complete rounds/30 K4 trials/47,602 actions; PPO and reset each
have six rounds/36 K4 trials, with 57,192 and 57,304 actions. All frozen
recoveries have no pending trials and matching source/learner counters. The
append-only corpus remains live; enumerate and hash only committed records
inside each frozen source cursor. Copying this prefix adds no robot actions.
