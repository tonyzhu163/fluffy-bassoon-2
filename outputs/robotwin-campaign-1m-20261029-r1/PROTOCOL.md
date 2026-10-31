# Shared one-million-action RoboTwin campaign

2026-10-29 Asia/Shanghai. The human's confirmed decision in “Plan VLA
in-context learning” caps RoboTwin at **1,000,000 executed robot-control
actions total**, shared across methods, fresh PPO rollouts, source validation,
native/target evaluation, settling/noops, retries and qualification. LIBERO has
its own independent1Mcap. This new pilot replaces the next-launch10.24M-per-
learner proposal; all original runs and contracts are preserved.

Use the same frozen released Motus pi0.5 motor, H32, public frozen VLM plus
proprioception features, learned per-chunk initial noise, six VLRLA source
tasks and six controller-held-out target tasks. Motor training covered all50
tasks; report transfer to new steering tasks, not whole-system unseen tasks.
Train K4 same-task trials; evaluate K8 and episode indices1/2/4/8, with5–8
explicitly longer-than-training trials. Uniform shuffled source-task blocks
are unchanged. Keep successful and failed episodes.

## Allocation within the cap

| Allocation | Maximum executed actions |
| --- | ---: |
| Feature-conditioned DSRL-style SAC source |272,000|
| Persistent GTrXL/PPO source |272,000|
| Separately trained episode-reset GTrXL/PPO source |272,000|
| Source validation: all6tasks xK4 x3methods x400 |28,800|
| Target:6tasks xK8 x6conditions x400 |115,200|
| Native reference:1episode/task before and after target |4,800|
| Already executed r4source and camera qualification |13,026|
| Remaining retry/settling/qualification reserve |22,174|
| **Total ceiling** |**1,000,000**|

Target conditions: persistent PPO normal/memory-reset/reward-blind, SAC
frozen/online, and trained-reset PPO normal. The latter's extra reset lesion
is redundant. Source validation uses frozen parameters, fresh seeds at or
above3,000,000,000 disjoint from the source sampler's seed range, K4context,
and no target SGD. Actual costs can be lower after early success. Reallocate
no unused reserve silently into source training; report all phase totals.

## Pilot training and stored experience

PPO uses **8 complete K4 trials/update** to produce a learning curve within
this budget, explicitly departing from the original256-trial VLRLA-scale
proposal. The final source budget cut is recorded and bootstrapped. All PPO
sequences come from the current recurrent behavior policy; SAC replay is not
substituted for PPO rollouts. Keep original PPO objective/optimizer settings.
SAC retains warmup500chunks,batch256,UTD20 and its declared bounded-uniform
warmup. Neither arm updates the VLA weights. An 8-trial batch contains at most
12,800actions, suggesting at least about21 source updates if episodes run to
the limit; actual success can yield more trials/updates. This remains a small
pilot, not a reproduction or convergence claim.

New collection saves lossless raw decision-time main/wrist RGB and
proprioception before/after each chunk, language/task/trial/seed/reset identity,
sampled noise, issued joint-action targets and exact executed-action mask,
raw/chunk rewards, true terminations versus truncations/durations, behavior
version and immutable policy snapshot. PPO records pre-squash samples,
behavior log-probabilities, values, recurrent history linkage/reset masks and
clock inputs. Keep full replay, all optimizers, samplers, RNG, pending PPO
trials and coordinator recovery. Original r4raw RGB was not saved and cannot
be reconstructed from its feature/hash artifacts; it costs13,026prior actions
with the gate but receives no new raw-corpus source credit.

A single benchmark journal reserves control-action intent before simulation,
resolves the exact executed count afterward and conservatively charges an
unresolved intent in full after a crash. It imports immutable prior receipts
idempotently. All runs/retries share this ledger; source per-method caps and
the total cap both apply. Never overwrite an active worktree or old output.

## Runtime and next compute decision

The optional demo_clean final-camera collector passed the paired gate at
source8114dd2 on two scenes: all observations/features/actions/counts/outcomes
matched, including400-action timeout and117-action success; paired speedup
2.1785x, about26.96actions/second. Extrapolation to the remaining~987kactions
is **about10.2GPU-hours of collection alone**, excluding optimizers, raw I/O,
all-task variability and startup. Allow roughly10–14hours before measuring
the added learning and corpus overhead; no full-run compute authorization is
implied by the interaction cap.

Budget/corpus/driver integration checks passed:84tests in12.67seconds on
Baige job-ytyeswlw54wg, source065777ceccf1c52a69d7def92399e35774b4bddd.
The test allocation ended23:27:28 Shanghai after1m10s; conservative combined
initial compute accounting is42m10s, below45minutes. Raw images are roughly
24GB uncompressed at this action cap, plus about20GB worst-case immutable
SAC acting snapshots; reserve60GB working headroom before collection. The proposed next training
allocation is a **two-hour, one-GPU resumable slice**, with the same ledger and
source recipe, full recoveries at boundaries, and actual optimizer/throughput
measurements. Launch that materially larger time budget only after user
authorization per AGENTS.md; continue necessary code/review work meanwhile.
