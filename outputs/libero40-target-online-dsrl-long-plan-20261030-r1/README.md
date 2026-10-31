# Longer target-online DSRL development continuation

Prepared executable plan; root owns remaining32h reconciliation, the two
6,000s component holds and actual deployment. This creates no allocation or
action reservation. Preserve the terminal six-case parent and all its files.

The six-case onset run (`ba4674f`, `libero40-target-online-dsrl-six-20261030-r1`)
completed all0/1/4/8/13/16-episode queries with **4/6 successes at every point**.
Tasks0/40 failed every frozen query and all16 training episodes, despite3,200
optimizer steps each. Task30 remained successful and reached800 steps.
Tasks10/20/50 remained successful without passing500 replay chunks; their
queries still used the published pre-update native Gaussian behavior. The
unchanged aggregate score establishes no learned gain. It also does not
establish convergence or a negative result at the paper's roughly10k online
sample scale. This paragraph is the canonical-summary insertion for the
parent's concurrently edited `docs/frozen-vla-steering-summary.md`.

Continue only the two failing cases, explicitly selected using onset outcomes
as a **development diagnosis**, not an independently selected confirmatory
benchmark. Keep state0/seed123, the same frozen official OpenPI pi0.5 motor,
canonical image64/proprio8 SAC, batch256, UTD20, >500-chunk warmup and fixed
weights during each episode. Keep recorded4104D prefix features separate from
the SAC actor input. No source-pretraining admission for any target trace.

The existing `run_case` loop already supports arbitrary episode milestones.
Keep its original milestone prefix0/1/4/8/13/16 and append40/80/160/240.
The parent recovery's `next_milestone=6` stays byte-identical and now points to
40. Reuse the original16-episode frozen query; do not execute it again.

Migration occurs only inside the allocated driver, after the GPU probe and
focused Linux tests. Copy the selected case's entire raw traces, physical
journals, six milestone recoveries and final recovery. Preserve original
contract/index/recovery bytes under `parent-*`; change only the latest
recovery's explicit new source/contract/cap metadata. Hash every copied parent
file and all numerical learner/replay/optimizer/global+motor RNG/progress
leaves. The real allocated SAC restore also checks complete state equality
before its first new gradient step. Historical archives keep their old
contracts. Parent source/index/clock/recovery hashes are fixed in each new
contract. Exact-source retries share that worker's remaining6,000s clock;
failed and unresolved physical attempts remain charged.

Each parent case has9,030 already executed actions:6,560 training,2,460
queries,10 qualification. Its learner has16 episodes/640 chunks/3,200 updates.
Each worker gets at most100k **new** actions, so the reused journal's cumulative
cap is109,030. The two copied prefixes total18,060; unique target campaign
cost is parent35,558 **once** plus each worker's new charges. The original
parent GPU charge1,851s is also counted once, then all new worker attempts.

| New query after total train episodes | New actions/task if every episode lasts400 | Maximum actual replay chunks |
| --- | ---: | ---: |
|40|10,260|1,600|
|80|27,070|3,200|
|160|60,280|6,400|
|240|93,490|9,600|

These include new training, current/prior new query measurements and one
10-action qualification. Retries/uncertain RPCs consume the remaining6,510
actions. Shorter successful episodes reduce observed chunks; episode counts
do not guarantee10k samples. Even9,600 chunks with pi0.5/H10 is not a matched
reproduction of the paper's motor/horizon, task or released500k-SGD schedule.

Measured reset→optimizer-burst→trace times for failed400-step post-warmup
episodes14–16 are22.32/22.08/22.02s (task0) and20.58/20.27/21.26s (task40),
each800 updates. The224 remaining training episodes extrapolate to roughly
83/77min respectively before fresh JIT compilation, migration, queries and
larger replay/checkpoint I/O. Hence two6,000s ceilings are reasonable upper
holds, not a guarantee of reaching240. Two independent one-GPU workers can
run concurrently for at most100min each (3.33GPUh total). Stop with a complete
episode+optimizer recovery if the cap/time prevents another full episode;
the last completed query may then be160. Do not invent an extra6h recovery.

Root's reconciled original32h pool has12,626s available; the proposed two
6,000s holds leave626s unassigned. No new GPU-hour approval is implied or
requested. Each wrapper reserves60s scheduler allowance and stops its driver
before the outer timeout. Existing `run_case` headroom receives an extra180s
buffer for JIT/serialization. The parent remains terminal and immutable.

Deploy fresh worktrees with the reviewed exact commit, then root may run:

```sh
bash scripts/diagnostics/vla_target_online_dsrl_continue.sh \
  EXACT_SHA libero40-target-online-dsrl-long-20261030-r1-task00 \
  /mowenjie/fluffy-bassoon-2/outputs/libero40-target-online-dsrl-six-20261030-r1 0
bash scripts/diagnostics/vla_target_online_dsrl_continue.sh \
  EXACT_SHA libero40-target-online-dsrl-long-20261030-r1-task40 \
  /mowenjie/fluffy-bassoon-2/outputs/libero40-target-online-dsrl-six-20261030-r1 40
```

For an exact-source routine retry append `--resume` to that same command.
Both wrappers include actual allocated GPU and semantic checks. No actor or
simulator work runs during import, static preparation or development-host
analysis. The new tests are prepared for allocated Linux, not run on macOS.
