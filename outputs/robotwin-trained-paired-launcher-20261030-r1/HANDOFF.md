# Paired trained launcher — CPU qualified, awaiting owner handoff

The launch artifact is `c353970a316837b13277ec65587e58e5cb25ce56`.
Its allocated Linux CPU gate passed47tests in19.402seconds on Baige
`job-k1upj2ycy60q`; the combined owner-control / future-panel / paired gate
passed123tests with exit0. See `CPU-QUALIFICATION.json` and hash-verified
`linux-cpu-gate/` receipts. No trained GPU result is claimed.

The scientific executable remains `57016ba08eed79043fd67bf8b085bd6d42e6d766`.
This separate launch artifact starts the original persistent and episode-reset
PPO streams from their new owner-issued checkpoint manifests. It does not use
the fresh qualification controllers, create ledgers, submit allocations or
control the current serial writers. SAC remains under its existing owner.

The execution owner requested this preparation after reviewing57016. Only that
owner controls the drain, checkpoint audit, terminal release of each old
allocation and admission of the new exclusive allocation. A paused supervisor
or historical checkpoint alone is insufficient. Preserve current collectors
until the new artifact and its CPU gate are ready.

## Scope and cost

One new genuine N6 round per trained arm adds six complete K4 trials, one PPO
update and five optimizer steps toward the original100-round goal. Maximum
new source actions are10,800 per arm /21,600 combined. Original input-due
evaluation stages run normally before lazy source pools, reusing completed
units; their maximum remaining allowance is31,600 PPO /17,200 reset, bounded
by their original reservations. Full conditional maximum is70,400 actions.
A newly reached milestone after this one source round remains due at the
process boundary. No altered evaluation schedule or source-only override is
permitted.

Canonical worker caps stay1,226,400 PPO /1,168,800 reset, with1,100,000 source
safety actions each, bound to parent SHA
`1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0`.
No prior-cost imports, cap edits, copied journals, fresh learners or automatic
retries are allowed.

The first proposed allocation is one96GB GPU,32CPUs,192GiB RAM,16GiB shared
memory, independently limited to45minutes. The parent runtime includes CPU
admission, the one idle CUDA probe, both children, cleanup and CPU audit within
2520seconds; each driver has2160seconds with180seconds boundary reserve. The
outer timer is2650seconds with20seconds kill escalation, under the independent
2700-second scheduler cap. Collector group admission remains600seconds. If
input evaluation leaves insufficient time for a whole source group, the slice
must stop with its recovery and is not a successful paired-round result.

The source owner clarified that the human removed the old overall compute
ceiling and authorized continuing the original100-round source goal. Existing
13-hour allocations are operational slices, not a prepaid shared GPU pool.
The owner must issue a2700-GPU-second operational reservation for this exact
first slice, binding that authority, actual prior charges and remaining
physical allowances. No fictitious global compute balance is required. The
LIBERO32-hour budget and both closed RoboTwin qualification holds remain
unavailable.600seconds for the pair without due evaluation is an estimate
from the453-second fresh test, not measured trained speed.

## Input documents

The owner creates a fresh plan using `plan-template.json`. All `OWNER_*`
placeholders must be replaced. Each method supplies the new57016-bound
manifest, full coordinator checkpoint and predecessor frozen-motor receipt,
each with an absolute path and SHA256. Bind the actual immutable parent at
`/mowenjie/fluffy-bassoon-2/campaigns/robotwin-native-wholetrial-20261030-r2/manifest.json`.

`owner_release` references JSON with schema
`robotwin_trained_pair_owner_release_v1`, `runtime_commit`57016, the fixed
`parent_manifest_sha256`, and `methods` containing `ppo` and `ppo_reset`.
Each row has `released:true`, exact `allocation_id`, `terminal_state`, a
hash-bound `terminal_receipt`, identical `manifest` and `checkpoint` refs,
and `canonical_head` with `journal_events`, `journal_sha256`, `charged_actions`
and `decision_count`. The source owner verifies scheduler terminal state and
process identity before issuing this receipt; the launcher does not control
or infer release from an old immutable file.

`compute_binding` references JSON with schema
`robotwin_trained_pair_compute_v1`, original `campaign_id`,
`original_total_compute_ceiling_seconds:null`, `source_goal_rounds:100`,
`admitted_scheduler_gpu_seconds:2700`, `maximum_scheduler_gpu_seconds:2700`,
`qualification_capacity_reused:false`, exact plan `resources`, identical
`owner_release` ref, hash-bound `authority` and
`prior_charge_and_physical_snapshot` refs. These bind original goal-completion
authority and this new bounded operational reservation; they cannot reopen a
closed qualification hold. Actual scheduler charges, including startup and
failure time, must be reconciled after terminal state. No automatic larger
slice follows a pass.

Compile the completed input plan on CPU with the pinned launcher. Compilation
only constructs the exact argv and validates its structure; it does not assert
live ownership, load a checkpoint or run CUDA:

```bash
"$PYTHON" "$LAUNCHER/scripts/diagnostics/robotwin_trained_pair.py" \
  --launcher-commit "$LAUNCHER_SHA" \
  --plan "$INPUT_PLAN" --plan-sha256 "$INPUT_SHA" \
  --compile-plan "$COMPILED_PLAN"
```

Retain the printed compiled SHA256. In the independently capped exclusive
allocation, the single invocation is:

```bash
bash "$LAUNCHER/scripts/diagnostics/robotwin_trained_pair.sh" \
  "$LAUNCHER_SHA" "$RUN_ID" "$COMPILED_PLAN" "$COMPILED_SHA"
```

`LAUNCHER` is a clean detached checkout at the final launch-artifact commit;
the plan's `worktree` is a separate clean immutable checkout at57016. Deploy
through Git/bundle, verify both exact commits, and review current main before
the new run. Do not modify any active worktree.

## Gate and result

The source owner's allocated Linux CPU job ran all four new targeted test
files with `CUDA_VISIBLE_DEVICES=''` and the qualified environment:47passed,
zero failures/errors/skips. This includes the genuine tiny-CPU-Adam trained
N6 continuation through the final corpus/ledger/ACK audit. No computation or
tests ran on macOS; static bytecode and shell syntax checks were the local
checks. The combined gate additionally passed45owner-control and31future
evaluation-wrapper tests. It executed zero live source controls, robot/VLA
actions or production SGD.

Before CUDA, an external90-second CPU subprocess checks both new manifests,
complete trained Adam/config/sampler/RNG/task state, original parent/asset/LR
receipts, full unchanged canonical headers, exclusive locks, journal chains,
owner-release heads, zero unresolved work, remaining whole-unit allowances
and input evaluation cursors. It rehashes the bindings and checks both locked
heads remain stable. It never constructs a `CampaignBudget`. A filesystem
lock RPC previously stalled despite nonblocking flags; timeout/unreaped helper
state therefore blocks CUDA. The hard allocation bound is still required for
uninterruptible kernel/filesystem failures.

After that gate, run one fresh strict-idle GPU/CUDA probe and two independent
drivers on its single attested UUID. A separate guardian tracks owned Linux
sessions and PID start times, including descendants in different process
groups. Parent exit/EOF triggers bounded cleanup. This is covered by the
prepared real-process CPU tests, not a claim that arbitrary guardian/container
failure is recoverable. Telemetry aborts above85GiB or90% used memory, retaining
the memory trace and exact outcome; the fresh test peaked65.55GiB.

Success requires normal child exits, no owned sessions left, released GPU
contexts, and a separate CPU audit of +6trials/+1update/+5SGD per trained arm,
original2e-4/1e-5 replay tolerances and KL<=0.1, complete raw RGB/decision/native
mask counts equal to canonical physical charges, unchanged motor, durable
group ACK and full new checkpoint/manifest, preserved old completed units and
correct input/newly-due evaluation cursors. `audit.json` names the next
hash-bound manifests. Any missing/failed/uncertain result requires owner audit;
there is no automatic retry, refund, new ledger or next allocation.

This artifact does not establish a trained speedup, long-run native stability,
or a fix for the existing native GIL/OIDN issue. Scaling beyond the first
reviewed paired round follows its actual result and owner-controlled handoff.
