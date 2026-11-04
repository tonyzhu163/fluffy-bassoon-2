# TurnOffStove replacement: 50-case native qualification

**Status 2026-11-04: all 100 native episodes and the independent Linux CPU
video/action audit completed.** Baige `job-ltf38050md7y` succeeded and released
both GPUs at13:07:44 Shanghai. Faucet18/50 and Kettle32/50 first successes;
the predeclared rule selects Faucet. No new PPO source is launched.
Execution owner: Codex chat `01a1a4de`, on the human's request to run this plan.
[Execution and accounting](EXECUTION.json). TurnOnMicrowave stays OOD.

## Question

TurnOffStove was at 7/50 native and 205→204/1,280 TRAIN first successes. It
gave the PPO critic almost no signal (explained variance ≈0.13). Which
replacement has native first success in the 5–70% PPO headroom range **at the
PPO training horizon of 600 controls**?

The atomic sweep's 20-case estimates are coarse:

| Task | 20-case first success | Wilson 95% |
| --- | --- | --- |
| TurnOnSinkFaucet | 10/20 at native horizon 600 | 29.9–70.1% |
| TurnOnElectricKettle | 8/20 at native horizon 450 | 21.9–61.3% |

## Run

- **Policy and collector:** unchanged released pi0.5 Human300, the same
  native lane collector and the same checks as the
  [atomic sweep](../robocasa-atomic-baseline-sweep-20261103-r1/PLAN.md).
  This means three cameras, H50 chunks, 10 flow steps and a deterministic zero
  residual, with every executed chunk equal to its native base chunk. No SGD.
  [EVAL.py](EVAL.py) and [QUALIFY.py](QUALIFY.py) differ from the sweep only
  in the task list, case count, horizon, seed banks and worker count.
- **Tasks:** both at horizon **600**. TurnOnElectricKettle is registered at
  600 only inside this process; its native registry horizon is 450. For it,
  the readout also counts first successes at or before step 450, so the
  result can be compared with its native rate.
- **Cases:** 50 per task. Native seeds are `730100 + 1000·task_index + slot`:
  730100–730149 for Faucet and 731100–731149 for Kettle. These are disjoint
  from the atomic sweep, the portfolio banks, PPO TRAIN and the checkpoint
  DEV/TEST banks. Motor seeds are
  `172000000 + 100000·task_index + 1000·slot + chunk`, above every sweep
  motor seed. The CPU check asserts both disjointness rules.
- **Resources:** 2 GPUs × 4 lanes. Rank 0 runs Faucet and rank 1 runs
  Kettle. 100 episodes / 60,000 controls.
- **Budget:** process deadline 2,700 s and scheduler limit 3,000 s, so at most
  6,000 GPU-s (1.67 GPU-h), with no automatic retry. Expected ≈0.7 GPU-h from
  the sweep's throughput.

[COMMAND.sh](COMMAND.sh) is the sweep's launcher. Changes:
- 2 GPUs instead of 8;
- scaled CPU/RAM/SHM floors: 32 CPUs, 256 GiB, 32 GiB;
- this run's paths and deadlines.

Before launch, run `QUALIFY.py <code> <output>` on allocated Linux CPU against
the native registry. Record the exact `main` commit, GPU probe and job ID.

## Decision rule (fixed before data)

First success at 600 controls, out of 50:

1. **TurnOnSinkFaucet** if it is 3–35 (5–70%).
2. Otherwise **TurnOnElectricKettle** if it is 3–35.
3. Otherwise neither; report back.

The rule prefers Faucet because:
- it has the higher screen rate;
- its native horizon equals the PPO horizon;
- a button-press Kettle task would make the OOD TurnOnMicrowave transfer
  nearly in-distribution.

Report both tasks' first success, success at end and Wilson intervals
whichever is chosen. A new 10M PPO source for the chosen task (≈75 GPU-h)
needs separate human approval.
