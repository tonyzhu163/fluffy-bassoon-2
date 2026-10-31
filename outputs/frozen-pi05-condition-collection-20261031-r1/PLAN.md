# Source-condition collector: prepared component

This is the condition/actual RL collection component of the human-authorized
`frozen-pi05-endtoend-20261031-r1` pilot. It creates no simulator, raw corpus
writer, scheduler, monitor or action ledger. The existing execution owner /
`finish_assembly` alone supplies allocation, deployment, the canonical budget,
recovery storage and the common complete-episode runner. This is a new bounded
feature-SAC pilot, separate from old frozen-VLA runs and the DSRL/native-PPO work.

Status: source prepared; allocated Linux qualification and actual A1 selection
are pending. No physical teacher, source dataset, improved behavior, qualified
DPT teacher or student result is claimed by this component.

## Fixed experiment contract

The complete pilot ceiling is 8 GPU-hours, 300,000 all-in physical actions,
64 GiB output and 1,800 aggregate CPU-worker-seconds for fresh qualification.
Phase action subcaps are A1 30,000, source 120,000, source validation 30,000,
condition crossover 15,000 and student evaluation 105,000. Every reset policy
action, control, failure and support counts in the same existing ledger. No
subcap, maximum episode count or unused old budget grants additional capacity.

Natural A1 selects one of `place_phone_stand` / `place_object_stand`, its actual
native horizon and a qualified four-coefficient mapping. The calibration
positive control does not choose the physical interface to make its inverse
work, and cannot replace natural A1 reachability evidence. Bind actual motor,
interface, selected mapping, range/units, task-horizon and source-selection
receipts before constructing `SourceProfile`. Target57 is excluded.

The two hidden conditions use the *actual physical mapping's* constant mode:
`delta = arm_bound*tanh(.4*B[0])`, sign minus/plus. B[0] has support on arm
indices 0,1,7,8; grippers6/13 are zero. The post-steering, pre-execution callback
adds signed delta then clips to verified joint ranges. It never modifies reset,
motor weights, instruction or initial public observation. Both conditions share
one static corpus contract; switch only between complete trials. Actual reset
hash pairing, command cancellation residual/clipping and real controller-by-
condition transfer remain physical qualification requirements. Sensor dynamics
can reveal calibration, so reward-causal necessity is not assumed.

Only the action mapping has candidate inverse coefficients
`[-sign*.4,0,0,0]`. On noise/KV, an external command inverse is not a valid
four-coefficient DPT label. Same-mapping source search or SAC must qualify it.
Command cancellation uses declared roundoff tolerance and unchanged grippers;
it does not assert bitwise physical trajectory identity.

Six independent `NoiseSAC` lineages comprise two TRAIN and one held-out VAL per
condition, with the exact method keys in `PROFILE.json`. Their profile is
2062 public features, action4/bound1, batch32, warmup32 valid chunks and UTD4.
Old SAC defaults stay unchanged. Initial collection is 24 complete episodes
per lineage, balanced round robin; 48 is only a later upper bound under actual
remaining resources. All actions remain bounded-uniform until the first
*completed* optimizer cycle. Updates occur only after a complete episode;
chunks beyond warmup earn exactly four cycles each, with no warmup debt.

Before/mid/after validation uses two fixed fresh cases per lineage at episodes
0/12/24: 36 validation episodes in total. An admitted continuation may add
36/48 milestones. Restore a frozen actor/private RNG; the callback must use
bounded-uniform behavior for update0, never an untrained actor. The collector
checks that validation did not alter source learner/replay/RNG and requires a
committed receipt containing both actual episode IDs. Save failures too.

A2 crossover is 4 fresh scenes x 2 conditions x 3 controllers =24 episodes:
the two primary TRAIN teachers and ONE preselected context-blind controller.
Do not choose a different blind control per condition or filter to oracle
successes. Fixed initial validation behavior supplies the stationary-retry
reference; no extra unbudgeted matrix is implied. At horizon600, initial source,
validation and crossover maxima are86,400/21,600/14,400 actions, each subordinate
to the phase/global/GPU/storage ceilings; actual candidate horizons need proof.

AD qualification needs measured actual behavior improvement beyond stationary
retries/clock effects. Flat and failed lineages remain in the corpus. DPT has a
separate actual same-mapping teacher competence gate; a physically qualified
fixed inverse/search policy can label DPT even if SAC behavior stays flat.
Use the same predeclared second TRAIN teacher for a condition's TRAIN and VAL
query labels (or the same separately qualified oracle); held-out VAL behavior
is not a different label teacher. Competence includes the actual label sampling
mode and pure policy/coefficients digest. No targets select teachers or models.

## Integration and bounded recovery

Driver dependency: `8248b1cefe8757b5e43f2a1b6b503e5b93831659`. Use its
`run_complete_episode`, `CoefficientSteering`, canonical raw writer and supplied
`CampaignBudget` unchanged. `collect_round_robin` calls that runner with actual
lineage `spec.method`; no duplicate environment loop exists here.

Teacher callbacks supplied by the execution owner:

- `guard()` enforces actual remaining GPU/CPU/storage clocks and reserves
  checkpoint/cleanup headroom before work; it cannot silently reset clocks.
- `checkpoint(event,payload,immutable=bool)` saves the full state atomically,
  verifies it, and returns `{path,sha256,retention}`. `retention` is `immutable`
  for protected initial/milestone/published-prefix/final artifacts and `rolling`
  for bounded latest recovery. Keep prior latest until replacement verifies;
  preserve the pre-cycle and interrupted states if an optimizer cycle fails.
  Prune only unreferenced ordinary recoveries after dependency checks. Never
  publish a mutable latest path as an immutable dataset reference.
  `BoundedTeacherRecovery` supplies this adapter using the existing atomic
  corpus primitives and Torch save format: two rolling files, protected
  immutable versions, hash-checked loads, private/full RNG and control snapshots.
  Create a fresh exclusively owned recovery subdirectory per lineage/job;
  recover from a hash-bound previous artifact rather than reopening a writer
  on an active/ended run. Loaded global/control state is reconciled/restored by
  the execution owner, not silently applied by a helper.
- `archive_actor(learner)` uses the existing pure-actor archive identity,
  parameter digest and `sac:update:<cycles>:<digest>` version; do not retain
  duplicate full replay checkpoints for every decision. Archive only once per
  actual acting version, even if queried repeatedly within an episode.
- `validate_source(immutable_checkpoint_ref,two_fixed_specs)` executes the
  frozen source-only panel in the existing runner/ledger and returns a committed
  `{path,sha256,complete:true,phase:'source_validation',episode_ids:[...]}`.
- `publish_prefix(record)` binds complete episode manifests/reset records,
  immutable initial/final full recoveries and validation receipts. Compatible
  useful TRAIN prefixes may be fitted while collection continues, within the
  same funded phase. This callback does not grant a fit or skip data gates.

Full recovery contains actor/critics/targets/temperature, all three optimizers,
replay/position, private interaction and training RNGs, counters, pending action,
episode and optimizer-cycle debt, completed canonical episode manifests and
validation receipts. Partial physical episodes and partial optimizer cycles
are preserved and block silent fresh reset/replacement. The execution owner
must reconcile them; no fictitious simulator restoration is promised here.

Estimate actual full-state bytes x retained immutable versions plus two rolling
slots per lineage, actor archives, raw corpus, students and all temporary output
before teacher launch. Also admit checkpoint write volume/time inside the
finite clock. Stop cleanly at whole-episode boundaries if the remainder cannot
cover all work;64 GiB is an aggregate ceiling, not per worker.

Student2076 input uses the ONE driver-owned causal projection:
`control.student_observation(native2062)` appends its previous14 actual valid
agent-requested commands. Driver typed `student_previous_agent_command` is
recorded before the next decision; every physical reset zeros it. Offline
exporter reads that typed field. `agent_requested_actions`, `valid_action_mask`
and `last_valid_agent_request` bind it to the last *executed* slot, not an
unexecuted tail. No duplicate projection helper is provided. Teacher stays2062.
Latent condition, post-calibration actuator commands, clock, lineage and IDs
are audit-only; query labels never replace actual history or outcomes.

## Qualification handoff

Mac work is editing/AST/diff checking only. The focused tests are prepared,
not run here. Execute with the existing trusted runtime on allocated Linux,
inside the sole owner's aggregate1,800-second qualification allowance:

```
python -m pytest -q tests/test_steering_actuator_condition.py tests/test_steering_condition_collection.py --junitxml=<fresh-owned-output>/condition-collector.junit.xml
```

Retain actual job/probe, executable/import hashes, unique JUnit cases, teardown
and actual scheduler cost. The tests exercise actual small CPU SAC updates and
full recovery as well as synthetic command callbacks; they cannot qualify
native runtime, hidden-condition transfer, learning improvement or a DPT teacher.
