# One complete batch14 source PPO qualification

Prepared, not launched or reserved. Root may fund at most2,400 cumulative
GPU-seconds and20,000 simulator actions from the2,540s unassigned original32h
pool, leaving140s unassigned. The budget template is deliberately unfunded.
No full source collection,100-update run, reset arm or target evaluation is
authorized by this launcher.

Use the existing official frozen OpenPI215 pi0.5 LIBERO40 motor, receipt
`ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`, the qualified
cu128 student overlay, and the pinned LIBERO simulator. The canonical40-goal
inventory is `/mowenjie/fluffy-bassoon-2/outputs/libero40-source-1m-20261029-r1/schedule.json`:

- File SHA256: `a2873aafbe7d047a301a8b4f6cb51e5a72b79b0af55c40607d691c0148acaeb5`.
- Internal schedule SHA256: `a59bbbbb4b6050cd4a330d395347dd4edc6257b55beebd5c6a91e896355a0716`.
- Actual inventory40 source goals; train IDs0..39, validation40..49. No target trajectories.

The new native-horizon batch module uses this task inventory, not its older
400-step collection budget. Source horizons are Spatial220/Object280/Goal300/
LIBERO10=520, each with10 settling actions. Use fresh policy seed2, full memory,
width128/layers2/heads4/context300. One fixed behavior snapshot collects14
complete K4 trials before any optimizer step. Two epochs produce **two global
Adam steps**, learning rate1e-4, over all real decision positions. This is the
implemented global batch14 objective, not the earlier seven-trial-minibatch
proposal. GAE and recurrent state reset at each trial; feedback/context persist
across its four episode attempts.

The all40 source cycle interleaves suites by task number. This bounded first
batch visits14 goals: Spatial00..03, Object00..03, Goal00..02, LIBERO10 00..02,
state0 with the14 scene seeds drawn once from the declared sampler. It runs56
episodes. Actual state/vector hashes remain recorded; a seed is not a new
physical state. Never report this qualification as all40 trained exposure.

Worst episode/settling cost is18,400 actions; initial qualification adds10,
so the first uninterrupted batch fits at18,410. The20k cap leaves1,590 for
retry/qualification/unresolved intents. A lost long-trial prefix can prevent
completion; retain its charge and pending batch rather than silently expand
the cap. Parameter updates occur only after all14 complete trials are ready.

Recovery saves complete partial batches, the fixed behavior hash, optimizer,
sampler, Torch/Python/NumPy state, motor RNG and task cursor after every full
trial. A ready batch is committed before its optimizer burst, so retrying an
interrupted burst does not recollect interactions. Raw per-RPC public inputs,
features/noises/decoded actions and outcomes now supplement complete-episode
PPO traces, retaining evidence when a current episode is interrupted.

Static review found no new GAE/replay/trial-identity defect. One setup recovery
gap was fixed: if qualification fails before `learner/latest.pt` exists,
`--resume` may initialize the same seeded learner only after proving there are
zero known/uncertain primitive actions and no raw training traces. Setup
artifacts and all settling charges survive. A missing learner checkpoint after
any training action is rejected. Existing complete/partial/ready-batch tests
remain the actual allocated qualification; the added test covers this real
setup-recovery edge rather than the shell launcher.

Use one GPU with the existing qualified16CPU/96GiB allocation settings. The
launcher uses the Python3.10 probe/test environment with **only worktree/src**;
the Python3.11 cu128 overlay is added only for the model runtime. Focused batch,
native-task, identity, conservative-cost and worker-clock tests precede models;
the motor RNG round trip runs in the actual OpenPI environment.

The cumulative worker clock includes setup, tests, failed attempts, teardown
and60s scheduler allowance. The first slice is2,340s. The trainer's absolute
soft stop is300s before the outer timeout, accommodating a long native K4
trial/checkpoint without granting additional time. An insufficient slice may
finish with a preserved partial or ready batch; that is an honest runtime
outcome. Exact-source retries share the same2,400s clock and20k action cap.

For capacity planning, record per-trial collection seconds and committed batch
collection time, replay/update times and Torch exact replay/update peak memory.
A1Hz NVML trace of the **allocated UUID only** records sampled total device
memory including OpenPI, rendering and PPO. The final qualification receipt
includes actual cumulative time, conservative actions, full recovery hash and
both kinds of peak measurements. The sampled NVML peak is not an exact
instantaneous maximum. Use actual task lengths and reset costs when preparing
the later1M-per-learner estimate; this one update establishes neither robot
performance nor a memory benefit, and is not a horizon-matched batch-size
performance comparison against the legacy400-step runs.

## Later frozen evaluation compatibility

`learner/candidate.pt` already uses the existing `frozen_vla_rl2_policy_v1`
export format. Its manifest binds the batch contract, completed update count,
counters and pending-trial count. `load_rl2_policy` and `TargetPPOController`
can execute those network weights unchanged:4104 input features,32D bounded
latent, same frozen motor interface. No network/exporter rewrite is needed.

The production source-validation/target launchers currently call
`vla_source40_frozen_validation.load_models/source_admission`, which correctly
require legacy `version:1`, `openpi_pi05_source40_shared_ppo_v1`,172k budgets
and successful R1→R2→R3 ancestry. They intentionally **reject** the new batch
recovery format `frozen_vla_rl2_complete_trial_batch_v1`.

The later adapter must separately admit that new format: bind candidate
SHA/manifest to `latest.pt`, its `contract.source` native-batch14 provenance,
`progress.iteration/counters`, complete task cursor, optimizer/sampler/global
and motor RNG, actual coverage and physical/time ledger. It must distinguish
pending collected trials from completed optimizer exposure and bind a fixed
source update before deployment. A separately trained episode-reset arm is
required before the existing two-model full/trained-reset comparison can be
used. Keep the legacy R3 admission unchanged. This one-update/14-goal
qualification is insufficient for the production all40/source-validation
gate, and this plan schedules no target episodes.

After root creates a funded record from `budget-template.json` and deploys a
fresh clean worktree, the exact launch inside its allocation is:

```sh
bash scripts/diagnostics/vla_source40_batch14_qualification.sh \
  EXACT_SHA libero40-batch14-source-qualification-20261031-r1 \
  /absolute/root-funded-budget.json
```

An exact-source routine retry appends `--resume` to that same command. Neither
the helper nor wrapper submits a scheduler job or creates a canonical action
reservation. All active jobs and older source artifacts remain unchanged.
