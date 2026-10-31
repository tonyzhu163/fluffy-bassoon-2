# Per-episode noise: static preparation, not a launch grant

Prepared 2026-11-01 from main `03ed46840cefe1c9ada101ae0fdb7cead42d4b44`. No models, simulator actions, tests or
allocations were run for this preparation. The learned positive-control gate
is still pending in the saved online-DSRL evidence. This document does not
change the completed 384-episode plan or spend its unused allowance.

## Reference admission before any interface GPU job

The original screen selected task 59. The separate online owner has a different
selection rule and case bank; its saved main records do not yet bind a selected
task, trained endpoint and qualified gain. Do not infer those from a chat's
untrained native/tied scores. The saved CPU continuation tests establish
software checks, not a learned positive control.

A new pre-action plan must bind the reference task, BDDL and physical case
hashes, exact official motor/READY hash, H10/execute10/10 flow steps,
400-action horizon plus 10 settling actions, reward, policy/checkpoint and
source hashes, SAC recipe and fixed endpoint, all initial controls, learning
curve, final controls, frozen-motor check and inclusive costs. It must show
learning beyond untrained tied noise and initialized SAC, rather than only a
native-sampling gain. Report the declared persistence criterion and all
negative outcomes. If replication fails, diagnose warmup successes, replay,
critic/actor updates and fixed panels before proposing interface collection.

Use an existing continuation only after that review passes. If its task or
case contract differs from task 59, document the reconciliation explicitly;
this is not completion of the original task-59 Stage B. Never pool the two
screens or call their overlapping development cases unseen queries.

## First conditional E1: timing at fixed 32 dimensions

Keep the motor, task, physical reset, noise distribution, dimension and lack of
observation conditioning fixed. Compare these two **untrained search** arms:

- `chunk32`: draw a new unbounded 32D standard Gaussian at every H10 chunk,
  tiled over the 10 flow-noise positions, as in DSRL warmup.
- `episode32`: use that candidate's first 32D Gaussian at every chunk of the
  entire episode, with the same H10 tiling.

This is a timing contrast. The qualified learned DSRL policy is a separate
positive-control anchor, not one of these two matched search arms. Comparing
that reactive learner with a constant vector would change conditioning too.

Before collection, inspect this chat's saved task-59 episode records and bind
the lowest three state IDs in 1–8 for which both native Stage A draws failed.
Freeze their state-vector and observation hashes in the new plan. If fewer
than three meet the rule, use fewer; do not add states or inspect query 40–49.
If a reconciled reference requires another task, bind an equivalent prospective
failure-case rule and its existing evidence before actions; do not silently
apply task-59 failure labels elsewhere.

Freeze exactly three candidates with generator seeds **11013200, 11013201,
11013202**. Materialize and hash the 32D vectors and per-chunk schedules on
allocated Linux before actions, pinning generator/version/dtype. In each
paired timing arm the first draw must be bit-identical; only advancing the
candidate RNG at later chunks differs. Repeat each candidate three times with
motor seeds **11013300, 11013301, 11013302** and scene/reset seed **123**. Reset
candidate schedules at every repeat; do not adapt candidates to outcomes.
Store all candidates, seeds, schedules, traces and failures. A case is reached
when at least one frozen candidate succeeds in at least 2/3 repeats. Report
reach per case and arm, with no impossibility claim for a negative search.

## Prospective inclusive cost, to be bound by a separate grant

| Component | Maximum episodes | Maximum actions at 410 each |
| --- | ---: | ---: |
| 3 cases × 3 candidates × 3 repeats × 2 timing arms | 54 | 22,140 |
| Qualified learned DSRL anchor: 3 cases × 3 repeats | 9 | 3,690 |
| Matched native: 3 cases × 3 repeats | 9 | 3,690 |
| Actual-allocation qualification episode | 1 | 410 |
| Total | 73 | 29,930 |

The fresh grant must cap **all** startup, qualification, controls, search,
failures and cleanup at **30,000 actions / 7,200 inclusive GPU-seconds / one
GPU / 8 GiB output**, without automatic retries or borrowing. Stop before an
RPC without conservative capacity for its physical intent; unresolved intents
remain charged. No additional simulator qualification episodes fit this
schedule unless other planned episodes are removed prospectively. Time and
storage admission can stop the run early. Require a clean exact source,
allocated expected-count probe, same-noise parity, real simulator episode and
motor before/after signatures. Use existing journal and cumulative worker
clock semantics; do not wrap the old screen as a new stage.

## Implementation and qualification still required

The current `vla_headroom_screen.screen_episode` draws tied noise inside its
per-chunk chooser. The older `vla_noise_headroom` constant-arm pilot is a
source-task experiment with a different plan and is not a drop-in target
runner. A separate runner must freeze candidate artifacts, enforce the
predeclared episode identities, record applied noise for every chunk, reject
queries/wrong physical hashes, and preserve conservative intents and cumulative
clock on interruption. Test those mechanisms only on allocated Linux, with
fake simulator/motor coverage before the GPU gate. No implementation or tests
are claimed complete here.

Only after reading this 32D timing result, prepare a fresh bounded 4D plan and
grant with an explicit fixed basis and matched timing. E2 admits only E1
survivors at equal actions against the qualified reference. E3 needs declared,
qualified hidden conditions and matched/static controls after learnability.
The candidate/repeat seeds above are search identities, not extra learner
seeds. All records remain target development exposure, excluded from source40
and meta-pretraining. No broad factorial, motor/benchmark change or source
collection is authorized by this preparation.
