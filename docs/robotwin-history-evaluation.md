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
4. Open a separately admitted ledger with the whole panel cap and its exact
   plan contract. Pass the raw `RobotwinNoiseEnv`, not a `CampaignEnv` wrapper:
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
