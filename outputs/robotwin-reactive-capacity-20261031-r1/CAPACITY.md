# Bounded full-reactive capacity candidate

Prepared31October2026. This is an executable candidate and static source review,
not a runtime-qualified fit or a new resource grant. Latest reviewed main before
freeze: cc03ad217139237800809294afaf0a1c288927ea. No active or ended worktree
is changed. The sole source parent owns all CPU/GPU scheduling and observation.

The actual fixed4D A1 result is negative under its declared rescue rule. Its
17,591actions and1,770GPU-seconds remain charged once. This candidate diagnoses
the larger learned reactive KV/physical-head classes; a success would not admit
the failed fixed4D collector, condition task, AD/DPT labels, or ICRL claim.
There is no full32-case efficacy panel, new source learner or target task.

## Data and meaning

Use the exact published phone clean50 archive SHA745a9830…/255,542,392bytes,
revision3dc3b798668feb99ac61cc9086d84cbcc3d79186. The actual allocated stage0
inventory SHAe60cb3bd863bb6d8a014aadad652d64bb1834535f8e9cff6d53476fcd840102b
binds the already extracted12 whole HDF5 files. Canonical episodes0–7 are TRAIN,
8–11 VAL, selected before windows/outcomes. No new download, pickle loading,
trajectory truncation, outcome sorting, or scene selection occurs.

AUTHOR-CONVENTION.json preserves exact pinned RoboTwin source bytes/excerpts:
the collector filters plan_success/check_success seeds, replays seed.txt and
asserts success after collection. HDF5 explicit success flags remain absent;
they are never synthesized. get_obs records simultaneous3RGB views and robot
drive targets, not measured joint positions. The explicit target is the same
frame's recorded absolute14D drive target followed by future recorded frames,
offset0. Images use head/left/right camera order and JPEG→RGB. Dense recording
uses save_freq15 plus boundary images. The pinned source is evidence of the
author convention, not an embedded generator revision in those HDF5 files.

Runtime gates therefore remain essential: the exact loaded motor's official
AlohaInputs inverse-action, arm-delta, normalization, padding and decoder must
roundtrip each used physical target. State normalization must equal the actual
native observation path. All12 source trajectories are replayed, in original
order, as explicit absolute expert commands through the canonical recorder;
actual seed and initial drive targets must agree. Any failed replay stops before
fitting. This verifies executability in the deployed simulator rather than
equating published collection provenance with successful native deployment.
It does not assert equal simulation wall-clock cadence. Replay is separately
identified expert execution, never a bounded native-policy correction.

Targets are split into nonoverlapping H32 windows after whole-episode splitting;
only repeated terminal padding is masked. The original full HDF5 remains intact.
The actual prompt path is retained when applying input_transform, so actions
cannot disappear through the already-tokenized input shortcut. Official
roundtrip is checked against all14 physical coordinates.

## Fit and execution

Reuse exact Motus/RLinf/RoboTwin/native H32/five-flow-step scalar runtime,
public2062 current-observation features and canonical whole-episode runner.
No second simulator loop, loader, action normalizer or scheduler is added.

- KV: existing ReactiveKVResidual, last native layer, rank4, scale0.1,
  zero output bases; differentiate through the actual frozen suffix. Native
  cache/features are detached and never edited in place. No prefix/base update.
- Physical: existing width64 reactive head, same features plus base proposal
  and slot index; source-bound arm delta bound. Enforce remaining12-arm joint
  headroom. Native arm violations fail closed when safe zero parity cannot be
  preserved. Gripper deltas are exactly0; raw native grippers, including actual
  overshoots of[0,1], are preserved bitwise and are never clamped/repaired.
- Each: one seed20261031,100AdamW updates, global batch2/microbatch1,
  learning-rate0.001, no weight decay, gradient clip1. Whole-source-VAL loss at
  0/25/50/75/100 selects the checkpoint. The untrained checkpoint is eligible
  and its iteration is reported. Parameter counts, update losses, peak allocated
  CUDA bytes, elapsed time and complete optimizer/sampler/private/global RNG
  recoveries are retained. A timeout/failure preserves the prior finite recovery;
  it does not trigger automatic restart.

Flow matching uses pinned compute_loss interpolation/time distribution and its
32-coordinate per-element target, with explicit valid-time mean. Reports retain
physical14 and padded18 errors separately; the physical head reports12arm/2gripper
errors and unattainable target error under its declared bounds. Four fixed TRAIN
anchors and all whole-VAL windows monitor fitting at the same checkpoints.
An underfit, unreachable-target or clock-limited negative cannot reject the full
reactive class. This fixed schedule is not a convergence guarantee or sweep.

One charged case7 episode supplies actual native/zero-KV/zero-action parity,
frozen-parameter/cache checks and a nonzero actual suffix-to-KV gradient with
no motor gradients. There are then12 teacher replays and4 predeclared old phone
development cases0–3 ×native/KV/action. Each development case retains its original
actual reset seed/instruction/raw-observation/public-feature identity, with one
common fresh motor seed across its3branches. These previously opened cases yield
descriptive wins/losses/ties and completion actions only; no heldout/significance,
teacher-expertise or memory/adaptation conclusion follows.

## One residual unit and pending admission

At most25complete400-step episodes=10,000new actions, inside the remaining
12,409A1 actions. No episode is cut to hit a quota. No unspent actions admit extra
trials. Same canonical300,000-action ledger/30,000A1 cap and old header remain;
no imported prior per child or replacement ledger. All prior episodes stay intact.

Maximum new allocation5,430GPU-seconds, including cold setup/probe, qualification,
all replays, offline cache, both100-update fits, tiny controls, failures and
cleanup; prior1,770+5,430=7,200A1 limit inside28,800phase GPU-seconds. Parent must
provide the actual scheduler start and one absolute deadline before construction,
with at least60seconds reserved for cleanup. A whole episode needs240seconds of
remaining clock before reset. These are caps, not measured completion promises.
Planning split: setup/qualification900s, replay900s, cache600s, KV1800s,
physical fit180s, dev600s, close150s=5,130s plus300s scheduler variation. Phases
share the original clock; no phase gets a new clock or extra allowance. Actual
fitting throughput/memory are unmeasured and must be reported, never assumed.
All saved outputs, caches and recoveries share the existing64GiB phase ceiling.

Before GPU admission, run only the new prepared CPU test file on actual allocated
0GPU Linux, ≤300worker-seconds within the existing1,383CPU-second remainder.
23cases are prepared, NONE run on macOS. They exercise actual metadata bindings,
whole-trajectory masks, transform roundtrip/retained prompt, differentiable KV
flow, arm headroom/raw-gripper parity, finite optimizer/private-RNG recoveries,
interrupted-update preservation, cached-byte tampering and residual admission.
They use synthetic tensors, no VLA weights/simulator/robot/source ledger.

```bash
python -m pytest -q tests/test_robotwin_reactive_capacity.py --junitxml=capacity.xml
```

The source parent must deploy the exact reviewed commit through Git, prepare
the SHA-bound demo manifest with the metadata-only helper below, verify actual
full clean source/pins/storage/fresh allocation, supply canonical admission and
actual one-GPU probe, then invoke the executable. No GPU allocation is admitted
by this candidate file. Existing task57/source jobs remain untouched.

```bash
python scripts/diagnostics/prepare_robotwin_reactive_capacity.py \
  --inventory /ABS/phone-stage0-cpu-artifacts/inventory.json \
  --inventory-sha256 e60cb3bd863bb6d8a014aadad652d64bb1834535f8e9cff6d53476fcd840102b \
  --author outputs/robotwin-reactive-capacity-20261031-r1/AUTHOR-CONVENTION.json \
  --author-runtime-path /ABS/FRESH-CODE/outputs/robotwin-reactive-capacity-20261031-r1/AUTHOR-CONVENTION.json \
  --motor-sha256 de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb \
  --norm-sha256 1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb \
  --output /ABS/FRESH-INPUT/demos.json

python scripts/diagnostics/robotwin_reactive_capacity.py \
  --scope /ABS/scope.json --scope-sha256 SHA \
  --allocation /ABS/actual-allocation.json --allocation-sha256 SHA \
  --interface /ABS/interface.json --interface-sha256 SHA \
  --joint-conventions /ABS/joint-conventions.json --joint-conventions-sha256 SHA \
  --demos /ABS/FRESH-INPUT/demos.json --demos-sha256 SHA \
  --original-result /ABS/A1-R2/result.json --original-result-sha256 SHA \
  --model-path /ABS/EXACT-MOTUS --norm-stats /ABS/EXACT-NORM.json \
  --robotwin-root /ABS/ROBOTWIN --rlinf-root /ABS/RLINF \
  --gpu-probe /ABS/actual-probe.json --source-commit EXACT-CANDIDATE \
  --run-id FRESH-ID --output /ABS/FRESH-OUTPUT
```

Allocation extension fields: mode=original_a1_full_reactive_capacity_v1,
run_id=the same fresh ID, capacity_action_bound=10000, capacity_fit_config=CONFIG
from the executable, prior_a1_gpu_seconds=1770, reserved_gpu_seconds≤5430,
scheduler_start_unix and driver_deadline_unix bound to the actual allocation.
All unchanged common admission/source/head/ledger bindings still apply.
No refresh of an old operational envelope or sealed credit is permitted.
