# Two-source-goal subset comparison

This is a bounded step toward the user-requested source/data-scale comparison,
before the pending full40 collection. It retains all40 as the final source
bank and full90/Plus/Pro as the targets. It does not replace that objective.

The completed one-goal run trained AD/DPT full/static25/100/500 updates on
Spatial00. A separately collected LIBERO10:00 training stream is already
available (809 chunks,7 complete cases,6,620 SAC updates,8,275 actions).
The next source-data scale adds that goal while keeping the motor, interface,
architectures, optimizer and target cases fixed. Source streams are selected
by their predeclared campaign task IDs, not by target evaluation outcomes.

## Missing independent source validation

Completed: Baige `job-aehuoen0j6a5`, exit0,36 episodes/32successes,
1,047chunks/11,400updates/10,714actions;9 actual nominal states40–48.
All36 traces, paired datasets and full recovery independently hashed.
[Receipt](../libero40-source-l00-validation-audit-20261030-r1/README.md).
The completed index SHA256 is
`6b49b3d642e3292d5fa9d18d190ebd7f2d6dbdad513ce9a715d8b9312fa08927`.
The launch contract below is retained as the original specification.

Run one `libero_10:00/validation` campaign slot using the existing qualified
collector and a fresh immutable identity:
`libero40-1m-sac-20261030-l00-validation-r2`. The statically prepared unused
original `libero40-1m-sac-20261029-l00-validation-r1` stays untouched.

- Maximum12,000 executed simulator actions, including settling, retries and
  uncertain execution intents, drawn from the existing96k SAC-validation
  allowance. No global action-cap increase or duplicate reservation.
- One allocated Baige GPU,16CPU/96GiB/8GiB shared memory,40-minute ceiling.
  Completed L00 training cost8,275 actions/644.68s; validation throughput is
  not assumed equal. Preserve complete recovery if time-limited.
- Validation nominal states40–49, disjoint from train0–39. The schedule
  repeats actual cases with new seeds; report actual state IDs/hashes
  separately from seeded cases. Four episodes/case, only start another case
  if its complete worst-case physical reserve fits. No LIBERO90/Plus/Pro data.
- Fresh task/partition learner and replay; published batch256/UTD20,
  Gaussian warmup until replay>500 query chunks, fixed behavior stage and
  actor within each episode. Save actual learning history, paired AD/DPT
  prefixes and complete policy/replay/optimizer/sampler/global/motor RNG.
- Require actual optimizer advancement and finite losses, not merely saved
  files. A source actor remains a candidate, not an established expert.

The reviewed current main is `acca706`; the fresh validation worktree uses the
exact parent commit recorded by its launch. The existing collector has not
changed since the qualified source streams. Actual allocation probe and Linux
checks run before rollout; no model/test computation on Mac or dev host.

```sh
bash scripts/diagnostics/vla_source40_campaign_collect.sh \
  <exact-reviewed-SHA> libero40-1m-sac-20261030-l00-validation-r2 \
  libero_10 0 validation
```

## Training and target comparison

After validation completes, assemble exactly Spatial00 and LIBERO10:00 with
independent whole-case source partitions and exact motor/feature/noise pins.
Reuse the existing four-controller subset trainer and fixed25/100/500 update
exports. No extra simulator actions are incurred by offline reuse. Do not
call the one-goal→two-goal comparison a full meta-training corpus or a final
negative method result. Optimization cost, source cost and deployment cost
remain separately reported. The training/evaluation launch must record its
own concrete GPU/action ceilings before execution.

The concrete offline training slice is one GPU/50 minutes, zero new simulator
actions, four500-update trajectories and immutable25/100/500 exports. Keep
128 width/two layers/four heads/context300/batch8/LR3e-4/window128/seed0 from
the one-goal run. Model computation is bounded to2700 cumulative seconds;
each controller retains its600-second ceiling and full resume state. One-goal
training measured1863.26s; a second goal can change overhead and outcomes.
The launcher requires the independently checked completed L00-validation
index hash and all four original source indices, qualifies source contracts
on allocated Linux, then calls the existing assembler/trainer:

```sh
bash scripts/diagnostics/vla_source40_two_goal_training.sh \
  <exact-reviewed-SHA> <fresh-training-id> <completed-L00-validation-index-SHA256>
```

Append `--resume` only for the exact same source/run/data/configuration and
cumulative ceilings. Existing actor snapshots and singular frozen-source-SAC
metadata keep Spatial00 first; this is not a joint two-goal SAC baseline.

The independent target-online DSRL driver is being prepared separately. It
will retain published SAC settings, fresh target learners, true warmup cost
and separately frozen query episodes. A short learning-onset comparison is
not converged DSRL. Neither target trajectories nor query outcomes may enter
these source assemblies.
