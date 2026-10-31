# Native frozen-history live evaluation adapter

`meta_rl.robotwin_history_evaluate` connects portable native supervised
policies to an already qualified `RobotwinNoiseEnv` and `CampaignBudget`.
It allocates no hardware, constructs no simulator and updates no policy.
The execution owner supplies runtime construction, allocation admission,
wall-time supervision and process lifetime. Experiment cases, artifact hashes
and resource proposals belong in the dated plan under `outputs/`.

## Admission and runtime

1. Read the checksum-pinned plan and run `admit_plan(plan)` on allocated CPU
   before constructing the motor. It loads all four fixed100-update portable
   models plus the source SAC teacher, checks their checkpoint and data hashes,
   architecture, seed and source-training contract, then releases them.
2. The execution owner integrates the adapter into a clean checkout containing
   the qualified native backend. The backend file SHA and complete scientific
   runtime contract must match the plan. Only allocation-specific render PCI is
   excluded from the runtime comparison; actual renderer binding is checked by
   the qualified backend. Final-camera capture and raw corpus recording are
   required. The older backend inherited by the trainer branch is rejected.
3. Run the allocation's GPU/CUDA check and the existing capture/runtime and
   explicit-noise parity checks. The first real native episode can count as
   the real-episode qualification and as part of the declared panel. All
   startup, model loading, parity inference, resets and retries consume metered
   runtime, even when they execute no physical policy actions.
4. Open a separately admitted ledger with total cap57,600, empty source caps,
   `phase_caps={"target": 57600}`, and
   `contract["evaluation_plan_sha256"]=canonical_sha(plan)`. The adapter checks
   this exact identity before a reset. Pass the raw `RobotwinNoiseEnv`, not a `CampaignEnv` wrapper:
   `run_trial` owns the single pre-dispatch reservation for each native chunk.
   A wrapper would duplicate physical accounting and fails backend identity.

The backend must retain native Motus weights, H32/full1024 noise, native
action transforms, source-bound feature normalization, task-native horizons,
assets, synchronous scene ownership and CUDA/render-device binding. The
adapter never edits the backend or a source collector.

## Executing one reviewed unit

The execution owner's driver supplies these existing objects:

```python
from fluffy_bassoon.meta_rl.robotwin_history_evaluate import (
    admit_plan, load_agent, run_trial,
)

admission = admit_plan(plan)  # Allocated CPU, before any motor construction.
# Owner constructs qualified raw env, raw corpus writer and separate ledger.
agent = load_agent(plan, condition, device="cuda")
result = run_trial(
    env, budget, agent, plan,
    condition=condition, task=task,
    output=fresh_unit_output, run_id=allocation_run_id,
    references=native_result,  # None only for the first native Gaussian unit.
)
```

Native Gaussian runs first for each task. Every later condition requires its
complete native K8 result bound to the SAME plan hash. Requested seed, actual
accepted seed, raw initial-observation hash and public feature hash must match
episode by episode before a policy motor query. Prior native-before/after
single episodes or a different case bank cannot supply this receipt.

All students use the predeclared deterministic clipped physical mean. SAC
uses its deterministic physical `1.5*tanh(mean)` labeling convention. Native
Gaussian is the motor's UNCLIPPED standard Gaussian; the separate native-zero
condition supplies zero initial noise. Applying the steering bound to native
Gaussian would change that reference.

Each condition has one K8 trial per task. Full student memory and actual
noise/reward feedback persist across its eight episodes. Same-weight reset
controls erase recurrent memory AND previous noise/reward/boundary at each
physical episode, while preserving the observable episode clock. This is an
inference lesion of the fixed full model; it is not a separately trained reset
student. Independently trained static models remove feedback and recurrence
at each decision. Episode clocks retain the original K4 clamping convention.

## Accounting, failure and interpretation

Before beginning a trial, require the complete K8 action allowance. Every
chunk reserves its full requested native duration before motor dispatch,
then resolves the ledger with the actual valid executed duration. A replayed
intent or smaller reservation never authorizes a new motor call. An unknown
native outcome leaves its reservation for conservative forensic accounting.
No episode is shortened to manufacture a complete K8 result or hit a cap.

The adapter records reset intents/cases, completed episode files, native
renderer record hashes and scene-creation counts, frozen weight/motor checks,
actual physical actions and motor queries. Zero-action terminal outcomes are
retained as zero actions; no fictitious primitive step is added. Complete K8
units are the recovery boundary. A simulator fault preserves partial files,
raw corpus and charged/uncertain ledger events and requires diagnosis. The
adapter never retries an episode or reconstructs lost live physics state.

Report episode indices1/2/4/8, actual costs and the paired cases. Any optional
support/query description divides the EXISTING eight episodes; it adds no
support episodes. One K8 trial per target/condition is a small development
screen, with correlated episode outcomes. It neither selects models on target
results nor establishes ICRL or deterministic teacher expertise.

## Source100 AD500 isolated-query readout

The explicit `ad500-source100` live profile uses the same runtime, controller,
action reservation loop and fresh-process lifecycle. The legacy fixed100 and
source-teacher100 plan readers keep their separate immutable hashes. Its
prepared topology is in
`outputs/robotwin-ad500-readout-20261031-r1/plan.prepared.json`; this template
cannot run. A separately hashed bound copy may replace only `models`,
`fit_commit`, `train_index_sha256` and `validation_index_sha256` with actual
post-fit artifacts. Every other plan field must equal the pinned template.

The selector requires the source-only AD full/static `candidate-000500.pt`
endpoints, exact fixed500 recipe, teacher-free SAC100 AD/BC train/validation
metadata and index hashes. It rejects best-validation files, earlier endpoints,
partial banks and DPT labels. CPU artifact admission checks equal source
normalization across full/static. This admission measures no robot competence.

| Condition on each of two DEV tasks | Independent trials | Episodes |
| --- | --- | --- |
| Native, frozen sampled SAC100, AD full, AD static at K0 | Eight fresh queries per condition | 32 per task |
| AD full at K1, same full weights with episode reset at K1 | Eight fresh support/query pairs per condition | 32 per task |

The 12 condition/task workers each load one motor. Within each worker, every
K0 query starts a new controller trial; every K1 support/query pair starts a
new trial. Query feedback never enters another pair. K1 retains failed support
episodes. The paired reset lesion clears recurrent cache, previous noise,
previous reward and the carry flag at the query boundary, while both K1 arms
retain query episode clock1. It does not imply equality to K0 clock0. Static
retains its independently trained, per-decision feedback/recurrence removal.

The support environment bank is disjoint from the fixed query bank. Accepted
support seeds must also be disjoint from all accepted queries, and the reset
arm must match the full arm's support reset cases and complete support records.
Native queries establish accepted seeds/raw observations/public feature hashes
for all conditions. Mismatches stop before acting on the mismatched case.

Native noise remains unclipped Gaussian. Frozen SAC100 uses its sampled
physical tanh action, with a separate declared private action seed per query;
it does not restore the source teacher's K4 RNG stream. Students retain their
deterministic clipped physical mean. `reseed_action_rng` changes only the
action generator between episodes, preserving support memory and clocks.

Fresh execution is bounded by 128 episodes and 51,200 actions at400 steps,
zero SGD, no saved-control reuse, and a separately proposed5400 GPU-second cap.
K0/K1 complete-unit allowances are3200/6400 actions. Recovery uses only complete,
hash-checked units under the same plan/commit/cumulative ledger and clock.
These preparation caps grant no resources. This two-task K0/K1 development
readout does not fulfill the larger0/2/4/8 six-task adaptation protocol.
