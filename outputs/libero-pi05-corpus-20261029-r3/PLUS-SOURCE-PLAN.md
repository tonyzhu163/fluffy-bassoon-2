# Approved clean versus clean + Plus source arms

- Two data arms: original-LIBERO clean source histories; the same clean
  histories plus a small, explicitly declared Plus source partition.
- Same exact RLinf130 motor, shared prefix coordinates, clean validation and
  final held-out Plus/Pro cases for both arms. No Pro source data. No final
  test outcome selects source cases. Earlier v1/v2 artifacts stay unchanged;
  this r3 declaration records the newly approved Plus training arm.
- Existing Plus development cases **667 / 290** remain evaluation-only. They
  are not repurposed as training data. Their classification IDs are 668 / 291.

## Source cases fixed before their outcomes

| Runtime index | Classification ID | Category | Official suffix | Published difficulty |
| --- | --- | --- | --- | --- |
| 2387 | 2388 | Light Conditions | `light_11` | 5 |
| 1438 | 1439 | Sensor Noise | `view_0_0_100_0_0_initstate_0_noise_1` | 2 |

- Both extend `pick_up_the_cream_cheese_and_place_it_in_the_basket` in
  `libero_object`; typed full-goal lineage matches original training task
  `libero_90:47`, not task 57. Exact hashes: [source-cases.json](source-cases.json).
- Deterministic selection: lowest classification ID for this base in each
  chosen category, using the pinned static index, before any source outcome.
- Initial Objects Layout candidate 1985 (`add_14`) was rejected on static
  feasibility alone: its official init file contains one state vector
  (shape 149), not three source scenes. Light Conditions replaces that
  category; it uses the unchanged official 50-state base bank. No easier
  outcome was sought; its published difficulty is 5.
- Pinned Plus source `4976dc30028e805ff8094b55501d532c48fec182`; classification
  checksum `faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6`.
  Classification IDs are one-based; order-0 runtime indices subtract one.

## Runnable bounded collection

- One allocated GPU, at most 45 minutes. Two source workers; states 0/1/2 per
  variant; K=4 complete episodes per trial: **six trials, 24 episodes, at most
  6720 executed primitive actions**. Settling and qualification resets counted
  separately. Horizon 280; settling 10; execute native chunks of ten.
- Fresh full ordinary DSRL/SAC initialization independently restored for each
  trial; batch 32, UTD20, warmup 0. These are declared technical source
  settings, not a claim that the final actor is competent or improves monotonically.
- Fixed RLinf130 motor and 4104-dimensional native prefix features. No older
  OpenPI motor/features, per-source drifting coordinates or fabricated histories.
- Save raw public observations, actual executed actions/noise/rewards,
  nominal-state and observation hashes, trajectories, complete actor/critic/
  optimizer/replay/sampler/RNG checkpoints after each episode.
- Publish each complete trial immediately: AD target = actual executed
  noise; DPT target = the trial's final candidate actor at each actual query
  observation. Original contexts and feedback stay unchanged. Candidate
  competence remains a separate empirical question.
- All exports are **train-only**; both local/global scene labels are `train`.
  Keep the clean arm's validation files identical for both candidates. Source
  interaction cost counted once, shared by the paired objectives.
- Atomic `index.json` points only to fully written immutable trial exports;
  partial raw episodes/checkpoints survive failures. No resume execution or
  student performance is claimed merely from saved recovery artifacts.

Launch after integrating the exact clean collector/helper and motor sources:

```bash
bash scripts/diagnostics/vla_rlinf_plus_corpus.sh <exact-source-commit> <fresh-run-id>
```

The launcher runs allocation checks/tests, validates source case bytes, starts
isolated Plus workers, then calls `vla_rlinf_plus_corpus`. Parent orchestration
assembles the two data arms and evaluates frozen students separately. Final
test masks must exclude these source cases as well as the already reserved
development cases/family for **both** arms.
