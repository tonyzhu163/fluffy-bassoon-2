# Native VLA context adaptation: initial design plan

Date: 2026-10-28, Asia/Shanghai.
Status: design recorded; implementation and compute qualification pending.

The user endorsed exploring direct history conditioning inside a VLA, with
reward-driven latent optimization as a comparison. This record makes the
proposal concrete; it is not a training launch receipt or an ICML result.

## Source and scope

- Inspected local `main`: `5746cbb0db126ff01bb721da6b7560ffc47169b1`.
- Design worktree: `/Users/mowenjie/.codex/worktrees/native-vla-context/fluffy-bassoon-2`,
  created from `main`; the primary checkout remains on `main`.
- [Proposed method](../../docs/native-vla-context-icrl.md).
- Existing run contracts, datasets and active worktrees remain their own
  lineages. The coordination cutover is still governed by the repository's
  [coordination pointer](../coordination.md); this plan does not create a
  second live-status file or perform that cutover.

The first implementation milestone is a qualified native prefix-history
interface and a bounded controllability check. It does not include a multi-seed
training sweep, a new simulator task or real-robot deployment. No new resource
budget has been set and no job was launched for this design.

## Decisions

1. Start with explicit scalar history inside native prefix attention, current
   native visual observations, and LoRA on prefix attention. The base motor
   tensors and action expert stay frozen; adapters train before deployment.
2. Reuse the existing four-attempt native rotation contract for qualification.
   Keep its held-out theta/overlay split and its system-identification caveat.
3. Warm-start with conditional flow matching, then use complete-trial RL.
   Include direct RL and warm-start-only controls. Use a training-only value
   head; no deployment-time external learner is needed.
4. Treat a reward-essential hidden-choice task as a required next scientific
   gate, subject to motor competence and an explicitly bounded scope/budget.
5. Keep latent/prompt optimization as a comparison. First qualify whether its
   interface produces competent behavioral variation; prompt dimensionality
   and search budget follow that measurement rather than an arbitrary sweep.

## Reuse audit

| Existing component | Reuse | Required change or limitation |
| --- | --- | --- |
| `rl/maniskill_pi05.py` | Motor identity, native transforms, environment and action conventions | Preserve source/motor hashes and allocated execution checks |
| `meta_rl/pi05_bridge_train.py` | Trial sampling, reward timing and complete-trial return contract | New collector/replay for native prefix training; current driver owns a different policy |
| `meta_rl/bridge_distillation.py` | Verified decoder-aware inverse-rotation labels | Current history v1 omits observations and caches stale downstream features for this purpose |
| `meta_rl/pi05_warmstart_train.py` | Immutable manifests, split discipline and portable artifact ideas | Current loss trains an external bridge; not a native-context trainer |
| `rl/pi05_feature_bridge.py` | External comparison and feedback conventions | Do not repurpose its detached final-layer cache as a trainable native prefix |

## Ordered implementation checklist

1. Inspect the pinned installed RLinf implementation on allocated Linux:
   prefix construction, multimodal embeddings, attention masks, RoPE positions,
   suffix input gradients, flow-policy stochasticity and likelihood support.
   Record the exact source hash and selected adapter target modules.
2. Implement a separate history schema/input adapter with the same eleven
   public fields, chronological visibility, null-history bypass and explicit
   trial reset. Capture full public observations/actions for reconstructible
   training. Separate privileged labels/private plans from model inputs.
3. On allocated Linux, verify native empty-history parity, no future/trial
   leakage, frozen-base fingerprints and finite nonzero prefix gradients.
   Demonstrate tiny-set controllability, then live motor competence. Do not
   advance just because offline flow loss decreases.
4. Measure token counts, peak GPU memory, collection and replay time. Freeze a
   numerical pilot budget and validation thresholds in a fresh run plan.
5. Qualify on-policy likelihood replay before any PPO optimizer update. Rebuild
   changing attention representations; checkpoint optimizer, sampler and RNG.
6. Only then start a bounded warm-start/live-adaptation pilot, using freshly
   reviewed latest `main` and an immutable new run ID. Apply repository cluster
   availability, allocation and GPU/renderer admission rules at launch time.

## Evidence and validation for this design

The reuse audit comes from the local source, not a runtime experiment. In
particular, `make_dpt_record` saves detached final-layer K/V and suffix features,
and bridge replay intentionally does not rerun the motor. Those are appropriate
for its frozen-motor bridge but insufficient when prefix attention changes.

Documentation is checked for local link resolution and consistency with the
inspected source. No model computation or tests run on macOS. No trained
native-context checkpoint exists from this plan. A broader literature review,
including internal-context and test-time latent methods, remains necessary
before claiming novelty.
