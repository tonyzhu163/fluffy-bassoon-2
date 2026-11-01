# Native action-expert PPO: prepared, not launched

The primary experiment is an official-RLinf recipe-aligned two-iteration
`adjust_bottle` PPO pilot: released SFT fa8 initializer, frozen VLM, trained
action expert plus detached value head, H400, 256 environments × four rounds,
global2048/micro32/five epochs. It is a target-online qualification, excluded
from all source6/source40 meta-training banks. The full published25-iteration
history-generation schedule is **not** funded or reproduced here.

The new YAML copies the official c605 root configuration with explicit pilot
overrides; its nested env/model/FSDP defaults still come from the clean source.
Hydra forbids inheriting that root as a defaults child because it defines
`hydra.searchpath`. The upstream PPO implementation, GAE, masks, FlowSDE0.3 and optimizer are
unchanged. The new entry installs narrow worker/runner subclasses without
editing either upstream checkout. It synchronizes weights and calls the
official `evaluate()` before training and after iteration2 on the same128
fixed reset IDs; `runner.only_eval` is deliberately false. Raw initial public
observations, actual motor chunks, rewards/boundaries, valid masks, immutable
actor DCP and worker RNG/sampler sidecars make the comparison inspectable.
Matching IDs and matching observation bytes are reported separately.

Official demonstration documentation reports SFT85.94%→PPO96.09%; that is a
different published demo protocol (current official YAML defaults H200).
Our H400 result must be measured. VLRLA's bridge50.9%→63.4% is a separate
student comparison, not a PPO result. Exact paper initializer/seed list/code
match remains unverified; the released fa8 initializer is identified honestly.

## Execution and cost

Nominal training819,200 actions plus paired evaluation102,400 =921,600. A
single cumulative action-accounting root caps all ranks/attempts at1,000,000.
Training uses actual published `valid_action_mask` counts; evaluation reserves
full50-action chunks because upstream omits that mask in eval. Constructors
and resets also charge an explicitly conservative10-step allowance/scene,
not a claim of measured setup actions. Unresolved RPC intents remain fully
charged. If any ceiling/deadline interrupts an iteration, preserve its raw
prefix and the last completed optimizer boundary; do not invent an exact
mid-iteration physics resume or silently restart with fresh escrow.

GPU ceiling21,600 aggregate seconds includes startup, tests/probe, evaluation,
PPO, failure and teardown: eight GPUs ×45min or four ×90min. Four ranks retain
global2048 and micro32, using16 accumulation steps rather than8. In c605
`embodied_fsdp_actor_worker.py:610–668,835`, each microbatch loss is divided by
accumulation and FSDP averages ranks: every microbatch has coefficient1/64
under either placement. The official objective is an average of per32-sample
masked-token means, **not** a global valid-token denominator. Rank-local
shuffles/seed partitions and numerical reduction order can change; no bitwise
equivalence claim. Per-rank valid-mask and gradient receipts are saved.

The released optimizer defaults to constant LR, zero warmup, total_steps0
(`fsdp_model_manager.py:495–524`); the derivative makes those defaults
explicit. Stopping at2 therefore preserves the first two scheduler updates of
a25-step run. Actor Adam/value learning rates and all scientific batches are
unchanged. No scaled-down PPO is called a replication.

Fresh storage cap64GiB. Full DCP model/Adam/scheduler at0/1/2 is retained;
redundant full-weight exports are disabled. The disabled scaler state,
optimizer step counter, Python/NumPy/TorchCPU/CUDA RNG, rollout batch,
sampler generator/reset IDs/seed cursor and source trajectory cursor are
saved per rank. Train `auto_reset=False` and no prefetched bootstrap means a
completed-iteration resume starts by resetting the saved next source IDs.
After-baseline evaluation restores training RNG; after2 evaluation restores
the original eval RNG. Mid-SGD or mid-physics exact continuation is not claimed.

## Actual allocated commands

Use the latest reviewed combined source pin, **not** this document's branch
HEAD. The retained runtime is read-only:
`/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python`
(Python3.11.17, shared Torch2.11cu128/Ray2.59/OpenPI0.1.1/SAPIEN3.0.1,
own MPLib0.2.1). CPU compatibility remains pending an actual allocation.

```sh
# Actual0GPU/8CPU/64GiB; <=20min; no actor/env construction, no model inference.
bash scripts/diagnostics/rlinf_native_ppo_cpu.sh EXACT_SOURCE FRESH_CPU_RUN

# Root alone creates an approved funding document after reviewing CPU receipt.
# Actual4or8 GPUs on one node; explicit existing native assets path required.
export ROBOTWIN_ASSETS_PATH=/EXISTING/QUALIFIED/ROBOTWIN/ASSETS
bash scripts/diagnostics/rlinf_native_ppo_pilot.sh EXACT_SOURCE FRESH_GPU_RUN 8 FUNDING_JSON
```

Funding JSON schema (example is deliberately **unfunded**):

```json
{
  "format": "rlinf_native_ppo_pilot_funding_v1",
  "approved": false,
  "source_commit": "EXACT_COMBINED_SOURCE",
  "run_id": "FRESH_GPU_RUN",
  "gpu_count": 8,
  "maximum_gpu_seconds": 21600,
  "maximum_actions": 1000000,
  "additional_storage_bytes": 68719476736,
  "action_accounting_root": "/mowenjie/fluffy-bassoon-2/outputs/FRESH_GPU_RUN/action-accounting",
  "cpu_qualification": {"path": "CPU_ROOT/cpu-qualification-summary.json", "sha256": "ACTUAL_SHA"},
  "prior_attempts": []
}
```

Any retry/eval-existing invocation needs a fresh run/attempt identity, the same
world size/action root, and every prior terminal clock path/SHA in funding.
The remaining wall time subtracts all prior aggregate GPU charges; no retry
receives another21,600 seconds. A boundary recovery is selected only by the
latest complete saved boundary, never target success. Later extension to25
requires explicit new scope/resource review; this config remains capped at2.

Optional released-PPO evaluation uses the same official method in explicit
`NATIVE_PPO_MODE=evaluate_existing`, not the ignored only_eval flag. Its staged
input is `/mowenjie/fluffy-bassoon-2/inputs/pi05-robotwin-ppo-adjust-bottle-1223d8b9-20261031-r1`,
revision1223d8b9c387eb22c51350900d842d0f3b86cf14,
READY SHA52ebf3a2f84cba5317fbc0bd9bed1e5c37c338309bb532acda1e2dd9c2235a39.
It costs up to51,200 additional requested actions plus reset allowances, and
may run **only** if the same phase's remaining action/time escrow can reserve
that entire panel. It never gates the primary training pilot. Initial-state
IDs/observations and seeds remain inspectable across outputs; cross-process
public-observation equality must be checked, not assumed.

Local verification is limited to AST parsing, shell syntax and diff review.
Required allocated CPU tests cover real Hydra composition,4/8 global math,
scientific-drift rejection, unresolved physical/GPU retry charges, RNG round
trip, actual native runtime imports, explicit before/after ordering and the
environment-commit-before-checkpoint boundary. No Mac/dev tests or models ran.
