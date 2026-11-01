# LIBERO90 native-headroom screen for a pi0.5 DSRL replication

Drafted 2026-11-01 against main `940aef2`. **Plan only:** no implementation,
allocation or robot interaction yet. Next steps after replication are in
[NEXT-INTERFACES.md](NEXT-INTERFACES.md).

## Why

The published DSRL result is a gain from about 20% to 100% for a frozen motor
on a task it solves only sometimes. Our task57 run cannot show that: frozen
pi0.5 already scores 10/10 there, and the finished 500k-update curve stays at
10/10 with 14% shorter episodes
([result](../dsrl-pi05-task57-20261031-r1/RESULT.md)). Freezing the motor was
never the obstacle. **Headroom** is: a task where native pi0.5 succeeds
sometimes, so noise steering has successes to reach and amplify.

Saved evidence can't identify such a task. Each fixed100 LIBERO90 cell is one
execution on one state (state0/seed123), so it can't separate 0% from 30%.

## Candidates, fixed from saved records before any new outcome

Fixed100 K0 gives five same-case outcomes per task: native, AD full/static,
DPT full/static
([case results](../libero40-fixed100-target-plan-20261031-r1/case-results.csv)).
These differ in steering, so a mixed row indicates that noise changes the
outcome, the property DSRL needs. Rows with 0/5 or 5/5 are mostly
excluded.

| Source of candidacy | LIBERO90 task IDs |
| --- | --- |
| Mixed fixed100 K0 outcome (1–4 of 5) | 29, 33, 53, 59, 64, 70, 77, 79, 82 |
| Native flipped between two runs with matched inputs ([cross-run](../libero40-native-crossrun-20261030-r1/README.md)) | 38, 44 (33, 82 already listed) |
| Ordinary DSRL reached training successes (72/240, [long audit](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/outputs/libero40-target-online-dsrl-long-audit-20261031-r1/README.md)) | 0 |

That gives **12 tasks**: 0, 29, 33, 38, 44, 53, 59, 64, 70, 77, 79, 82. Task57 is
excluded (measured ceiling) and task40 too (0/240 under DSRL exploration).
The list is closed; no task is added after Stage A outcomes.

All are LIBERO90 target-suite tasks. Every screen record has role
`target-headroom-screen`, counts as development exposure, and never enters
source40 pretraining
([protocol](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/outputs/libero40-meta90-20261029-r1/PROTOCOL.md#target-roles)).
State0 (fixed100's target case) and states 40–49 (the DSRL query panel)
are not used.

## Stage A: native and tied-noise rates

Motor, runtime and observation: identical to task57. The official OpenPI
`pi05_libero` (receipt `ec064cce…d170`, parameter signature `c565dd3f…69e9`),
H10/execute10, ten flow steps, 400-primitive horizon, one shared motor per worker.

For each of the 12 tasks, on states 1–8:

- 2 native episodes, independent full-horizon H10×32 Gaussian noise;
- 2 tied-noise episodes, one 32D Gaussian draw tiled over the horizon, which
  is DSRL's warmup exploration distribution.

That is 16 native + 16 tied episodes per task and **384 episodes** in total.
Seeds are derived from (task, state, arm, repeat) and written to `plan.json`
before the first reset. Record actual initial-state and initial-observation
hashes, success, primitive steps and settling actions per episode, plus a
compressed raw trace.

The tied arm is required: DSRL learns only from warmup successes, so a task
whose native successes vanish under tied noise can't pass warmup even with
native headroom.

## Selection rule (fixed now)

A task is **eligible** if native successes are 2–9 of 16 (12.5–56%) and
tied-noise successes are at least 2 of 16.

Among eligible tasks, select the one whose native rate is closest to 20%. Break
ties by more tied-noise successes, then by the lower task ID. Record the next
eligible task as a runner-up. Do not run the replication on the runner-up
unless the primary fails for a recorded runtime reason.

If no task is eligible, stop and report the full 12×2 rate table. Widening
the bands, adding tasks, using other states or changing the motor is a new
plan, never a silent fallback.

Rates come from 16 correlated episodes (8 states × 2), so intervals are wide.
Report descriptive Wilson intervals per arm. This selects a development task;
it does not estimate population success rates. Runtime nondeterminism
(4/90 cross-run native flips) is one reason to use repeated draws.

## Stage B: replication run, separate admission

Reuse the finished task57 profile unchanged except the task ID: fresh
PixelSAC seed0, batch256, UTD20, warmup >500 chunks, training states 0–39
in cycle, query states 40–49, ten-episode panels after each 10k updates,
initial/final native panels, 500k-update stop. Query states 40–49 are not
seen during selection, so the selected task's query panel stays
outcome-blind.

Success criterion, fixed before Stage B: the final learned panel beats the
final native panel by ≥3/10 on the matched states and seeds, and the gain holds
in the last three learned panels. Matching the paper needs three seeds. One
seed is a pilot, and a second and third seed are a separate decision.

The task57 driver binds task 57 explicitly. Stage B needs a reviewed
parameterization of that guard, with the task57 contract and immutable run
untouched.

## Resources

| Stage | GPU | Simulator actions (all-in ceiling) | Storage |
| --- | --- | --- | --- |
| A screen | ≤3 GPU-hours | ≤170,000 (384 × 410 = 157,440 worst case + qualification/retries) | ≤20 GiB |
| B replication, one seed | ≤10 GPU-hours | ≤1,000,000 (task57 used 332,977) | ≤64 GiB |

Basis: fixed100 cost 7,126 GPU-s for 522 episodes; task57 used 28,686 GPU-s.
Each stage is within the standing ≤10 GPU-hour instruction, and each is its own
grant with no borrowing. Stage A runs on its own sources, not the
1M-action learner ledgers.

## Implementation gaps before Stage A

- `meta_rl.vla_noise_headroom` accepts only the old 78/10/10 train/validation
  split and one task, so it can't take LIBERO90 target tasks. Add a narrow
  multi-task screen profile that reuses the `OnlineSourceSimulator` reset
  path, `frozen_query`'s native-noise path and the tied-Gaussian warmup
  sampling of `run_episode` with updates disabled. Confirm these reuse points
  at the reviewed pin before coding.
- Focused CPU tests: seed derivation, arm schedule, closed task list,
  role tagging, selection rule on synthetic outcomes, and the admission refusal
  for state0/40–49 and unlisted tasks.
- On the allocation: GPU probe with `--expected-count`, motor parameter
  signature before/after, same-noise motor replay, and one real episode
  before the full plan.

## Outputs

`plan.json` (seeds, states, arms; hashed before rollouts), per-episode
`episodes.jsonl`, raw traces, `rates.csv` (task × arm successes/16 with
Wilson intervals), `SELECTION.md` (the rule applied, selected task, runner-up
or a stop), and a cost receipt.
