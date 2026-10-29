# Shared source40 PPO stream

`vla_source40_multitask_ppo` trains one recurrent noise policy with one optimizer
and one sampler across all40 original LIBERO source goals. It retains the
official frozen OpenPI motor, prefix-v2 features, and the raw on-policy records
described in [SOURCE40-PPO.md](SOURCE40-PPO.md).

Tasks follow a fixed round-robin list: Spatial0–9, Object0–9, Goal0–9,
LIBERO10:0–9. One complete K-attempt trial uses one task and one nominal
state/seed. Task choice never depends on outcome. Local training states0–39
are sampled by the existing full-trial PPO trainer; the task cursor advances
only at the first reset of a new complete trial. Memory starts empty at a
trial boundary and persists through its retry episodes in full mode.

At most one owned simulator subprocess exists. On a task switch, the driver
closes its exact owned process and launches the original fresh-reset LIBERO
server for the next goal. It checks source commit, task, BDDL hash, horizon,
state-bank count and settling protocol. A stable mutable metadata object
changes with the task, so the motor and feature encoder receive the current
instruction. This route prioritizes correct shared training; per-task startup
cost is measured separately and is not assumed negligible.

The case schedule is hash-bound and must contain all40 source task identities.
It supplies provenance; its SAC action reservations are not PPO reservations.
The PPO stream has its own explicit `--max-total-actions`, charged by the
physical journal for every qualification/reset/action, interrupted attempt and
unresolved RPC. Only complete worst-case K*(400+10)-action trials may start.
Source-trained weights and all optimizer/RNG/task cursor state persist across
goals. A run stopping before all40 have actual episodes reports its partial
coverage without claiming complete source exposure.

Each reset has an additional task receipt with the actual source task key,
local state/seed, BDDL hash and returned nominal-state/observation hashes.
`learner/latest.pt` atomically includes the task cursor, active task/case,
committed task exposure and motor RNG with policy/optimizer/sampler/RNG.
On resume the process requalifies first, charges that reset, then restores the
saved runtime so qualification cannot advance the source task schedule.
Interrupted trials may be recollected and remain physically charged. The
complete-trial learner recovery does not claim mid-episode simulator recovery.

The driver reuses the declared pilot PPO batch (one full trial/update) and
records its optimizer settings. It does not silently adopt VLRLA's much larger
parallel rollout batch. A separately trained reset controller needs its own
on-policy interaction reservation; its cost is not free reuse of full-memory
rollouts. Frozen validation on source states40–49 can use the single-task PPO
validation route with the shared final policy; validation never updates it.

Inside a qualified Linux allocation:

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_multitask_ppo \
  --schedule <hash-bound-all40-schedule> --output <fresh-output> \
  --checkpoint <official-pi05-libero> --openpi-root <qualified-openpi> \
  --simulator-python <qualified-simulator-python> \
  --simulator-pythonpath <explicit-simulator-module-paths> \
  --source-commit <exact-clean-commit> --expected-hostname <allocated-host> \
  --max-total-actions <PPO-reservation-within-the-shared-1M> \
  --max-elapsed-seconds <invocation-cap> --updates <requested-updates>
```

This source stream does not train on LIBERO90/Plus/Pro. Target support/query
evaluation and source validation consume their separately declared campaign
reservations. Neither rollout generation nor optimizer updates alone establish
high robotic performance, useful memory or sample efficiency.
