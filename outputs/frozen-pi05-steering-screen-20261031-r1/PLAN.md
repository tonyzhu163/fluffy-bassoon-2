# Frozen pi0.5: steering competence and useful histories

Prepared 2026-10-31 Asia/Shanghai for the user's requested continuation of the
redesign. Reviewed remote main `fd4a7c91207210951e5bbfc3ecb80c61fedd9052`.
This is a development-screen specification, not a completed experiment or a
new resource grant. Runtime admission requires a verified implementation,
an exact motor/data binding, measured capacity and a concrete resource receipt.

## Decision

Resolve two questions before more architecture or student-scaling sweeps:

1. Can a small controller improve the exact frozen motor on cases it fails?
2. Does experience reveal something that changes a useful later decision?

These are separate gates. Better reactive control supplies an interface;
it does not by itself supply an in-context RL task. Start with one benchmark,
one source task and feedforward controllers. Add persistent memory only after
both gates have evidence. Preserve the existing DSRL-first/native-PPO-later
execution order, its immutable runs and its separate budgets.

## Evidence that changes the earlier proposal

- VLRLA-bridge freezes both the VLM and action expert and adds a low-rank
  residual to the native KV cache. VLRLA-simp replaces the action expert.
  Neither is an additive residual on decoded robot commands. The paper's
  source PPO separately updates the action expert. Actual teacher actions
  can supervise a frozen bridge; the teacher's noise draws are not thereby
  valid noise labels for the original motor.
- The existing `Pi05FeatureBridge` reads native KV but changes flow velocity.
  It does not implement VLRLA's cache intervention. Its bounded tracker is
  an achieved diagnostic policy, not an upper bound on every residual policy.
- The saved ManiSkill instruction-swap experiment failed its explicit-name,
  swap-follow and per-object gap criteria. Prompt arms remain candidates,
  not a qualified primary interface. Its RL motor and the RoboTwin/official
  LIBERO motors are different checkpoints and cannot be pooled.
- Owner-reported task57 DSRL reached 10/10 query success, matching native;
  one saved panel used 1,270 versus 1,451 query simulator steps. This is a
  small completion-cost signal, not a success-rate or memory gain. The early
  training bank contained no failures. Task57 stays target-online only.
- Negative comparisons and small reward-blind gaps do not prove that frozen
  steering or feedback learning is impossible. Transition observations can
  themselves be informative; reward-specific claims need reward interventions.

Sources: [VLRLA protocol](../../docs/vlrla-protocol.md),
[existing feature bridge](../../docs/pi05-feature-bridge.md),
[instruction swap](../pi05-instruction-swap-thu2-20261029-r1/RESULTS.md),
[DSRL task57 plan](../dsrl-pi05-task57-20261031-r1/PLAN.md).
The [earlier diagnosis](../archive/icrl-headroom-diagnosis-20261031-r1/PLAN.md)
remains historical context; its universal oracle-ceiling, power-as-equivalence,
30% recovery and optimal-bandit requirements are not this screen's contract.

## First candidate bank and motor

Use RoboTwin source tasks `place_phone_stand` and `place_object_stand` as the
first candidate bank. Published source-only ALOHA clean demonstrations exist
in the [pinned manifest](../vlrla-protocol-20261029-r1/robotwin-datasets.json).
Their native-failure/teacher-success qualification is **not yet established**.
Do not infer native failure from poor SAC/PPO source validation.

The current backend is Motus pi0.5 through pinned RLinf, native H32, five flow
steps, three images and 14 decoded ALOHA joint targets. Use its exact checkpoint
digest, normalization digest, delta-arm/absolute-gripper inverse transforms,
asset revisions, task-specific horizon and native observation capture. Copy
these bindings from verified source-run receipts into admission; do not guess
them or substitute JAX OpenPI or a ManiSkill motor. No target task or target57
record enters task selection, fitting, normalization or teacher selection.

Select at most one of the two tasks using a separate development bank of eight
predeclared resets per task. Record every outcome. Reject a candidate if native
is at ceiling, neither controller can access a successful trajectory, or action
conventions cannot be verified. If neither qualifies, stop this pilot and report
that result before proposing a changed task distribution or motor.

Before fitting, lock a disjoint panel of 32 complete initial cases, their query
and motor seeds, and all comparison methods. Selection uses development cases;
the locked panel is never reduced to the native failures after evaluation.

## Minimal interface comparison

| Arm | Intervention | What trains |
| --- | --- | --- |
| Native | Original cache, sampler and decoder | Nothing |
| KV residual | Add masked, zero-initialized low-rank residuals to native KV; decode through the unchanged expert | Reactive cache adapter only |
| Physical-action residual | Add bounded corrections to decoded H32 by 14 joint targets before the existing execution path | Reactive residual head only |

Use the same current-observation information and base motor noise. The action
head may also read the base action proposal; disclose that input. Record each
controller's parameter count and computation rather than claiming architecture
matching. An existing source SAC controller may be reported as a separately
identified reference only if its exact motor, action/noise and observation
contracts match; its inclusion is fixed before the panel is opened.

For the KV arm, preserve token positions, layer/key/value layout and padding.
Zero output bases initially; do not append tokens or update prefix LoRA/base
parameters. A compact reactive adapter is an interface test, not a VLRLA
reproduction. Training recomputes the suffix through the edited cache with
autograd enabled for the adapter while motor parameters stay frozen. Saved
detached suffix/velocity records cannot train this intervention.

For the action arm, apply the correction in documented decoded command units.
Specify arm-joint and gripper bounds separately using source-only action data
and simulator conventions. Record base proposals, requested deltas, applied
deltas, clipping, executed commands and valid durations. Do not treat a 7D
ManiSkill Cartesian residual or normalized 32D padding as ALOHA joint targets.

Qualification requires zero-modification native parity on actual inputs/noise,
unchanged motor parameters, valid masks, real decoded-command parity and one
real episode. Gradient checks must establish a nonzero path into the KV output
bases through the frozen suffix, with no gradient into motor parameters. All
model computation and tests run on allocated Linux; macOS is for editing and
static inspection.

## Competent supervision before another online sweep

Begin with approximately eight successful source training trajectories and
four separate successful source validation trajectories, split by complete
trajectory before windows are made. Verify action transforms and actual
simulator executability; a published demonstration is not automatically an
executable teacher on a changed reset or embodiment.

Fit the reactive KV arm using supervised flow matching on correctly transformed
teacher action chunks. Fit the action arm to teacher command corrections from
the same observations and base proposal convention; report unreachable targets
under the declared residual bounds. Use source validation for checkpoint choice.
Record dataset size and updates; these tiny fits are capacity probes.

Evaluate native and both locked learned arms on all 32 paired cases, with
contemporaneous native controls and a predeclared repeated-control subset.
Use task/case-paired success differences, wins/losses/ties, completion actions,
total charged actions and intervals. Two planned learned-versus-native tests
use a declared familywise correction. A useful provisional candidate needs
at least a 15-point observed success gain and uncertainty excluding zero under
that analysis; otherwise report inconclusive or failure. This small panel has
limited power and one training seed; a pass justifies replication, not a broad
performance, sample-efficiency or ICRL claim.

If a physical residual looks useful, add a separately budgeted norm-matched
random-correction control before attributing the gain to learned correction.
Do not make exploratory model or task selections on the locked panel.

## Histories for a feedback-dependent task

After steering competence passes, construct a declared source distribution in
which a reward-relevant latent condition persists across attempts. Fresh scenes
may vary within the condition. The initial student observation/instruction must
not directly reveal that latent variable. Available behaviors must be physically
competent and their outcomes informative; verify both rather than assuming
uniform random choice succeeds with probability 1/k.

For a hidden-target task, balance reachable targets independently of object
position and appearance. Keep oracle target information out of student inputs.
Test matched contexts with different consistent reward labels and require the
next physical decision to change appropriately. Semantic prompts can be one
candidate channel only after explicit target-selection competence is qualified.

Two data routes have different requirements:

- **AD:** retain actual early, intermediate and late exploration from multiple
  independent teacher lineages, in update/interaction order. Keep failures and
  unsuccessful updates. Verify improvement on fixed separate source cases and
  beyond a stationary retry policy. Do not manufacture learning by sorting
  failures before successes or switching to easier tasks later.
- **DPT:** collect informative behavior contexts, then label only the query with
  a qualified teacher for the same frozen motor/interface and condition. Earlier
  actions and outcomes remain executed behavior. The context-generating teacher
  need not itself improve within every sequence. Qualified action targets can
  come from another action-generating teacher; noise labels need an explicit
  same-motor correspondence and cannot be inferred from its RNG seed.

PLD's base-policy probing followed by residual-specialist takeover is a useful
recovery-data design. Its within-episode correction trajectories alone do not
demonstrate learning across attempts. Keep that distinction in corpus reports.

Split whole tasks/conditions and teacher lineages before making context windows.
Retain raw current/next observations, instructions, motor and teacher identities,
base proposals, requested/applied interventions, actions, durations, rewards,
boundaries, chronology and source/target roles. Keep audit-only IDs and policy
versions out of student inputs. Store complete learner/replay/optimizer/sampler
and RNG recoveries separately. Reuse collection cost once without hiding each
new fit or evaluation cost.

## First adaptation test and stopping rules

Use one GRU learner with the winning interface before CTM/XL comparisons.
Report K0 and 1/2/4 support episodes followed by a separate reset query, charging
all support. Freeze every deployed weight. Pair full context, complete
cache-plus-feedback reset, reward blind, consistent reward shuffle and wrong-task
history with native, a trained static control and a budget-matched search/bandit.
Preserve scene/draw pairing and disclose any intervention that changes it.

Require evidence that informative feedback changes decisions and improves
queries over reset and stationary retries. Beating an optimal bandit is not
required to establish adaptation; matching a simple bandit limits the scientific
contribution. A 30% failure-then-success fraction alone is not an adaptation gate.

Stop at the earliest failed gate: unverified motor/data semantics, no competent
alternative, no reproducible closed-loop gain, no feedback-informative condition,
or no context benefit. Preserve negative results and complete recovery. Do not
launch a full factorial, scale the old successful-only corpus, revive the closed
RoboTwin exporter or treat unused historical budgets as a new allowance.

## Resource and ownership handoff

The saved-outcome audit and implementation specification add no robot actions.
Before a new GPU phase, the existing execution owner measures model memory,
fit speed, episode throughput and storage on allocated resources, then records
one all-in phase ceiling covering qualification, fitting, controls, evaluation,
failures, cleanup and recoveries. The native development screen and training
capacity probe are included in that ceiling; they are not uncharged preludes.
No new numeric GPU grant is created by this file. Honor existing standing
authorization where applicable; ask only for genuinely new scientific scope or
a materially larger budget. Never change an active run's contract in place.

Coordination roles: this chat owns the redesign record; the current LIBERO
owner owns task57 and its scheduler; the RoboTwin owner supplied backend/data
feasibility; the existing coordinator owns the independent saved-panel audit.
Each updates only its own row in `outputs/coordination.md`. No duplicate live
observer or heartbeat is needed for this planning work.

## Focused primary reading

- [VLRLA, section 4 and appendix A](https://arxiv.org/html/2610.19517v1):
  frozen KV bridge, distinct source-PPO histories and deployment protocol.
- [DSRL](https://arxiv.org/html/2506.15799v2): latent-noise steering of frozen
  diffusion policies, including pi0; not an automatic pi0.5 replication.
- [PLD](https://arxiv.org/html/2511.00091): frozen-base residual specialists
  and probing/recovery collection, principally pi0; not an ICRL result.
