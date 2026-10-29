# Shared all40 PPO pilot: prepared continuation plan

Status: executable preparation only. No source extension, reset-control run or
24-hour allocation was launched by this work. The user chose a1M simulator-action
cap per benchmark, shared by SAC, fresh PPO, validation, settling and retries.
GPU-hour expansion still follows AGENTS.md and the actual scheduler allocation.

The prepared full-controller architecture is width128/layers2/heads4/context300.
It requires a new bounded allocated qualification before full collection; the
measured width32 smoke is not promoted to the final controller by this plan.

## Qualification and measured scale

Baige `job-b7u5ogwy6udu`, source
`72bd91b988fa35acec324a6ea34cd2f427823120`, completed three shared-policy PPO
updates and six optimizer steps. Its twelve episodes visited Spatial00/01/02:
1,173 primitives +130 settling =1,303 total simulator actions,125 query chunks.
All twelve succeeded. Behavior likelihood replay error was0; approximate KL was
0.1323/0.0260/0.0370 with learning rate1e-4 and two PPO epochs. This qualifies
source execution and task switching, not an adaptation or generalization gain.

Saved update durations are18.67/30.09/31.81 seconds, totaling80.58 seconds.
The latter two include a task-server switch. Counting the corresponding
1,173+120 training/settling actions gives about16.05 actions/s during these
three update windows. Initial model/renderer qualification and scheduler/setup
are additional. Linear extrapolation at this easy-task rate is approximately
3.46 GPU-hours per200k-action worker, or6.92 GPU-hours for two workers. Other
task lengths, rendering, image compression, PPO dynamics and restarts can
change that rate; it is not an ETA or a timing guarantee.

Original raw source episode files total about28MB across this qualification.
The raw-only extrapolation is several GB per200k actions, not a1TiB requirement.
It excludes behavior checkpoints, full-trial tensors, optimizer recoveries,
journals and validation. A conservative provisional64GiB storage reservation
per worker covers substantially more than this observed footprint; measure it
again during the first production slice before treating it as a ceiling.

## Shared PPO action allocation

| PPO pool use | Reserved actions |
| --- | ---: |
| Earlier one-goal qualification | 9,500 |
| Shared task-switch qualification | 6,000 |
| Shared full-memory training | 200,000 |
| Separately trained episode-reset control | 200,000 |
| Frozen source validation, eight fixed goals ×two controllers ×4,000 | 64,000 |
| Unassigned PPO retry/qualification reserve | 10,500 |
| Total PPO pool | 490,000 |

Actual earlier qualification cost is3,107+1,303=4,410, but its original15,500
reservation remains retained. Unused reservations are not automatically freed
or counted as an extra data budget. New full/reset learners each receive the
same architecture, optimizer, source-task schedule and200k physical-action cap.
Each uses its own fresh on-policy interactions. AD/DPT reuse of the SAC corpus
is a separate offline comparison; it is not PPO replay.

The frozen validation slots can each fit up to two worst-case K4 trials
(2×4×410+10=3,290 actions) within4,000. They perform zero optimization and use
states40–49, separate from training states0–39. Target90/Plus/Pro support/query
evaluation is not silently charged as source validation or launched here.

## Exact launch and resume

Deploy the reviewed integrated commit into two fresh immutable run worktrees.
The following runs one worker inside an already authorized one-GPU allocation;
it does not submit a job. Example run IDs must first be checked unused.

```sh
bash scripts/diagnostics/vla_source40_multitask_ppo_collect.sh \
  <exact-SHA> libero40-pi05-ppo-full-20261029-r1 200000 100000 full 3600
bash scripts/diagnostics/vla_source40_multitask_ppo_collect.sh \
  <exact-SHA> libero40-pi05-ppo-reset-20261029-r1 200000 100000 episode 3600
```

The high optimizer target is a stop ceiling; action accounting is the intended
collection stop. Launcher defaults are width128, two layers, four heads,
context300, seed2, learning rate1e-4, two epochs and one complete K4 trial/update.
Named `--width`, `--layers`, `--heads` and `--context-length` arguments override
the architecture and are bound into its immutable worker contract. Architecture
defaults remain pending their own qualification. This pilot batch does not reproduce VLRLA's larger parallel
rollout batch. Architecture or schedule changes require a new reviewed run
contract and appropriate qualification.

For a width128 qualification, reserve a separate4,000-action slot within the
PPO pool and request three updates. That leaves70,500 actions after the two
200k training reservations and original15,500 qualification reservations; the
suggested64k frozen validation plus6,500 contingency then fit. This extra slot
is not assumed spent until actually reserved/launched. Width32 qualification
timings above cannot establish width128 update cost or memory capacity.

Continue the **same** source SHA, output, cap, target, mode and slice size:

```sh
bash scripts/diagnostics/vla_source40_multitask_ppo_collect.sh \
  <same-SHA> libero40-pi05-ppo-full-20261029-r1 200000 100000 full 3600 --resume
```

The launcher reserves the stream once in the shared PPO campaign ledger and
requires an exact ledger match on resume. `worker-time.json` binds the run's
source/configuration and retains per-attempt GPU probes, tests and logs.
Unfinished attempts remain charged at their full reserved slice plus60 seconds
of scheduler allowance. Finished attempts charge measured runtime plus the
same allowance; overruns are reported rather than truncated. The executable
upper bound is24 cumulative worker-hours, so two workers permit at most48
GPU-hours before validation. That is a protective planning ceiling, **not
authorization** to allocate48 GPU-hours. Actual approved worker/slice budgets
must be set in the scheduler and tracked by the owner.

## Coverage and recovery interpretation

Task choice is outcome-independent round robin, so completed trial counts
across tasks differ by at most one. Four retry episodes keep one task/state/seed.
Local state IDs are sampled uniformly with replacement from0–39. They are not
exhausted before reuse. At the observed approximately434 actions/trial,200k
actions allow roughly460 trials, around11–12 cases/task, with approximately10
distinct state IDs/task in expectation. At worst-case1,640 actions/trial only
about121 trials fit, aroundthree/task. Report actual visited IDs/hashes from
the reset receipts; **do not claim all40 state IDs per task are covered**.

Every qualification/reset/action is charged before execution; a new trial
starts only if its full worst-case reserve fits. The final short remainder may
stay unused. The full learner recovery atomically retains policy, optimizer,
sampler, Python/NumPy/Torch/CUDA RNG, motor RNG and task cursor at a complete
trial boundary. Resume requalifies and charges that reset before restoring the
cursor/RNG. Interrupted trials may be recollected, their old actions remain
charged, and uncommitted raw files are retained separately. Original runs and
checkpoints are never overwritten or migrated to a new code contract.

Before promotion, verify actual optimizer advancement, finite actor/critic
losses, task instruction/reset binding, action ceiling, full recovery, saved
behavior likelihoods and on-policy provenance. A saved candidate or high source
success rate does not establish useful memory, one-shot transfer or sample
efficiency. Evaluate those claims on separately accounted support/query trials.
