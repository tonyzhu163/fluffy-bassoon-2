# LIBERO-Gen history pilot

Owner: `libero_gen_pilot` sub-agent. Created 2026-11-11. Preparation began
from project main `4fad07f8baa281305214205c6b63597de42e15b7`; refresh and pin
the verified project commit before any accelerator launch. Additional budget:
**at most 5 GPU-hours**, only motor/interface qualification. No training or
full evaluation campaign is launched by this contract.

## Question and scope

Can a learned, frozen history reader infer a persistent hidden manipulation
rule from attempted actions and native outcomes, then choose the correct
action in a fresh scene? The motor is frozen too. The first setting adapts
the assets and task machinery of BPP's **LIBERO-Gen Spatial Combination**.
It is a new history protocol, not a reproduction of its published score.
Goal Chain is a later extension after motor reliability and feedback use
are established.

The binary deterministic-choice task is essentially one-bit task
identification, with noise from motor failures. It can demonstrate
feasibility and feedback use, but not multi-step evidence accumulation or
broad rule generalization. A later scientific campaign needs at least
three plausible hypotheses, rules that share candidate decisions in some
scenes, and held-out rule compositions. The five history conditions from
the toy study (demonstration, actions only, reward-bearing attempts, expert
correction, linked correction) are a later extension, outside this motor
qualification budget.

## One episode family

Use two identical black bowls and a plate as the placement target.
The published generator supports these assets and multiple bowl instances.
Candidate rules are, for example, “select the leftmost bowl” and “select
the bowl nearest the plate.” These are proposed custom relations, not
claims about released task names. Layouts and physical feasibility must
be qualified before locking this pair. Define leftmost in the fixed
agent-view image, and nearest by tabletop distance; the sampler may use
privileged geometry to enforce these relations, while the learned reader
must infer the observable relation from its images.

The exact first motor pair already exists in the release, both in
`libero_spatial_selected_combinations_inverse_view`:

| Candidate | Released task | Native goal |
| --- | --- | --- |
| Center bowl | `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | `On akita_black_bowl_1 plate_1` |
| Near-plate bowl | `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_2` | `On akita_black_bowl_2 plate_1` |

Their regions, fixtures, objects and initial predicates have exactly the
same static hash (`42805b832c40f5f2eaf4de48b3b3820e0bdbceafed790ab55e85fc45391e5c65`).
This supplies a concrete two-choice motor test without inventing unsupported
objects. Official Hub metadata confirms both HDF5 files exist, with sizes
and SHA256 values in `ASSET-MANIFEST.json`. Both files were downloaded,
SHA256 verified, and their 50-demonstration schemas passed CPU admission. The released init files have different byte hashes, so they must
not simply be assumed paired. Use an isolated init bank with the same
center-bowl task initialization copied under both candidate names, then
verify matching simulator state/RGB/proprio after reset. This is a custom
motor qualification, not the released benchmark evaluation. Successful
execution remains unverified. The
custom relation task uses the same two bowl-to-plate skills with qualified
layout variation; it is not asserted to be ready before simulator checks.

1. Sample one hidden rule, balanced across episodes, and keep it fixed.
2. Show one successful demonstration in which both rules select the same
   bowl. The full demonstration is deliberately insufficient to identify
   the rule; no truncated demonstration is assumed to work with BPP.
3. Allow two attempts on independently reset scenes where the rules can
   disagree; extend to four only after timing. Each attempt has RGB,
   wrist RGB, proprioception, selected candidate, executed trajectory,
   and the simulator's native terminal success bit.
4. Evaluate on a fresh query scene where the two rules choose different
   bowls, with the same hidden rule. Do not reveal the
   BDDL filename, object-of-interest field, hidden goal, or instruction.

For each scene, instantiate the hidden rule as the native BDDL goal, e.g.
`On(chosen_bowl, qualified_target)`. The same physical initialization must
be possible under either hidden rule. Object IDs, prompt ordering, scene
seeds, and filenames cannot identify the rule. Reject ties and collisions.
Alternative motor commands must both be executable from the same scene.

## Actual model interface

Official BPP source is pinned to
`58125e682acceefef3d60c071212dc483a4db1c9`; its LIBERO submodule is pinned to
`8f1084e3132a39270c3a13ebe37270a43ece2a01`.
The released Spatial Combination BPP checkpoint is demonstration-conditioned:
pair prompting, at most one full demonstration, RGB/wrist/proprioception
and action chunks. Its configuration ignores task language and goal image.
The rollout calls `policy.prompt(...)` before the episode; its rewards are
used for stopping/scoring, not passed into `predict_action(...)`.

Therefore the first learned system has two components:

- A small history reader trained offline on training-only histories, then
  frozen for evaluation. It consumes demonstration/current-scene features
  and `(scene, candidate, realized trajectory, reward)` history and selects
  a candidate skill. All scene features must be observable; privileged
  simulator positions are allowed only for a separately labeled diagnostic.
  Start with a small sequence model over frozen visual features and explicit
  candidate/action/outcome tokens. Train candidate-choice cross entropy
  using the known training rule to label the correct query choice; the rule
  itself is never an input. Train on balanced independent attempts, with
  varying context lengths including zero. This is offline meta-training of
  the reader, not online gradient RL and not a claim of zero-shot reward use
  by the released BPP. Architecture/training budget is set in the next
  contract after motor timing and available data are known.
- A qualified frozen motor. For the released BPP checkpoint, the selected
  skill is represented by a successful demonstration from a fixed skill
  library, supplied through its existing prompt API. The library includes
  both alternatives, and its selection never receives the hidden rule
  except in the explicitly informed reference. A language-conditioned
  checkpoint is a separately named fallback, not a BPP reproduction.

The learned reader is new work, not already supported by the released
checkpoint. A scripted/Bayesian reader is only a feasibility and information
reference. Neither network receives gradient updates during test histories.

## Comparisons and measurements

First compare one fixed learned reader on exactly paired histories from a
balanced, reward-independent candidate sampler: true rewards, masked rewards, shuffled rewards, and
demonstration only. Reward shuffling preserves the reward marginal but
breaks its association with the scene/action; report the actual fraction of
labels changed. Do not regenerate histories under each ablation. A
reward-aware collector's later action choices can leak the original rewards,
so its histories are reserved for a separate closed-loop experiment.
Match history lengths and remove success-dependent
termination/metadata leakage from the reader input. Report the distribution
shift induced by masking, and later use a matched training ablation if needed.

Add an informed reference that knows the rule but uses the **same motor**,
and a finite-rule inference reference with the same observable information.
Motor failures make negative rewards ambiguous, so report candidate-choice
accuracy separately from native physical success. Record whether the
requested candidate skill physically succeeded even when it was the wrong
hidden goal; this is diagnostic-only metadata, not a second reward channel.
Do not delete failed
executions post hoc. The informed motor is an empirical reference, not a
guaranteed mathematical upper bound.

Primary readout: paired fresh-query success gain from true feedback over
masked/shuffled feedback, versus history length. Also report choice accuracy,
repeat-error rate, motor success for each candidate, and latency. Split
training/validation/test scene seeds before collection: reserve seeds
0–9999 for training, 10000–10999 for validation, and 20000–20999 for test.
These are namespaces, not promised collection counts. Keep each complete
support/attempt/query history in one split, deduplicate physical initial
states, and select prompts/features/normalization using training data only.
Both binary rules appear during reader training; the test holds out scene
layouts, trajectories and episode combinations, not the rule vocabulary.
Initially claim only
transfer to fresh layouts; new-rule/compositional transfer needs a later
separately designed split. Size the campaign from measured episode time and
pilot uncertainty, not an unmeasured large grid.

## Qualification sequence and current status

1. Static release/interface audit and candidate inventory.
2. Separate BPP runtime, pinned code/checkpoint/data; do not modify shared
   legacy LIBERO environments or active worktrees.
3. One allocated GPU, in-step GPU probe and rendering-device binding,
   checkpoint load, one real simulator episode, then a small paired motor
   panel (the exact two candidates above × five scene seeds). Require all
   inputs/prompts to load, genuinely paired initial states, and at least
   four native skill successes out of five for **each** candidate before
   treating the interface as feasible. This is an engineering gate, not a
   precise motor success estimate. Expand to twenty seeds per candidate
   only if measured cost fits the remaining cap; otherwise report the
   uncertainty. Test both agreement and disagreement custom layouts before
   starting any reader training. Record all outcomes, wall
   time, GPU type, memory, hashes, and native reward traces. Budget accounting
   includes allocation startup and failures, capped at 5 GPU-hours total.
4. Proceed to learned-history collection/training only after the reader
   interface and each candidate's motor reliability are credible and a
   concrete next contract is written. Low motor success is a motor blocker,
   not evidence against in-context learning.

Current qualification status is maintained in [PREPARATION-STATUS.json](PREPARATION-STATUS.json).
At 17:28:55 Shanghai, `qual04` completed the native runtime smoke: both
candidate episodes succeeded, initial observations were exactly paired, and
frozen parameter versions were unchanged. Strict loading verified exact
equality of all 967 saved tensors on the allocated RTX4090. Raw receipts,
action/reward traces and videos are in [qual04](qual04/). Three earlier
startup failures are preserved. Finalized usage is 0.395 scheduler / 0.542
conservative GPU-hours of the 5-hour cap. The five-scene engineering panel
uses initialization indices 0–4; index0 repeats the smoke scene and is not
a fresh held-out test. The gate remains at least4/5 for each candidate.
Custom relational layouts and the learned history reader remain unqualified.

## Primary sources

- [Official LIBERO-Gen documentation](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/docs/liberogen.md)
- [BPP policy configuration](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/behavior_prompting/train_network/config/libero_policy_dunetp.yaml)
- [Observation and prompt schema](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/behavior_prompting/train_network/config/task/libero_defaults.yaml)
- [Actual rollout interface](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/behavior_prompting/train_network/env_runner/libero_image_runner.py)
- [Released Spatial Combination checkpoints](https://huggingface.co/austinpatel/bp_liberogen_spatial_combination)
