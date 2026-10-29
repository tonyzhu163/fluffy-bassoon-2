# LIBERO-Goal instruction-steering headroom (no training)

Dated plan, **2026-10-29**. Run identity `libero-goal-headroom-thu2-20261029-r1`,
secondary school cluster (thu2). Code: `src/fluffy_bassoon/rl/libero_goal.py`,
`scripts/diagnostics/libero_goal_headroom.py`, `tests/test_libero_goal_headroom.py`.

## Question

Is LIBERO-Goal a usable task family for a hidden-goal in-context RL paper in
which the instruction is vague, reward reveals the correct goal, and the
frozen VLA's language input is the behaviour to choose? Three checks:

1. A frozen pi0.5 succeeds with the explicit instruction.
2. Swapping the instruction on the same initial state and noise changes which
   goal is achieved (language actually steers it).
3. A vague instruction performs near chance.

## Environment and checkpoint

| Item | Value |
| --- | --- |
| Checkpoint | `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`, revision `6222623f635769bfc73c9472e29fab9b7fd8e027` |
| `model.safetensors` SHA256 | `4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683` (matches the Hub LFS hash) |
| Norm stats | `physical-intelligence/libero/norm_stats.json`, SHA256 `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1` (identical to `RLinf-Pi05-LIBERO-SFT`) |
| Model code | RLinf `e1a82eb` snapshot `fluffy-bassoon/cache/src/RLinf-e1a82eb` (tarball hash in the env setup receipt), task `eval`, `pi05_libero` |
| LIBERO | `rlinf-libero==0.1.3` (PyPI wheel SHA256 `6d66835128856cee77d3252dda902bf33c5478305670496ec5137abd05e770a5`), `robosuite==1.4.1`, `mujoco==3.3.7`, `bddl==3.6.0` |
| Assets | Original LIBERO assets (commit `8f1084e`, verified copy `fluffy-bassoon/cache/libero/assets`); the 851 files fetched from `RLinf/LIBERO-assets` before the mirror failed were byte-identical |
| Environment | `/cluster/nvme5/zhujiayang/fluffy-bassoon-2/envs/rlinf-pi05-libero-e1a82eb-20261029-r1` (uv, Python 3.11.16), replicated from `rlinf-pi05-maniskill-e1a82eb-r1` plus LIBERO; receipts in `envs/setup-rlinf-pi05-libero-e1a82eb-20261029-r1/` |

Why this checkpoint: RLinf's documented LIBERO pi0.5 checkpoints are the
few-shot `RLinf-Pi05-LIBERO-SFT` (pi_RL reports 77.1% four-suite average
before RL) and the full-data LIBERO-130 SFT used here. Full supervised data
over all 130 tasks gives the strongest language conditioning without RL
fine-tuning (the RL-tuned `RLinf-Pi05-PPO-LIBERO-130` was not chosen to avoid
reward-shaped instruction collapse). Both are public; no credentials were used.
They were downloaded through the public `hf-mirror.com` because thu2 cannot
reach `huggingface.co`; hashes were checked against `huggingface.co` metadata.

Inference follows RLinf's LIBERO pi0.5 evaluation: 256 px agent-view and
wrist images rotated 180 degrees, 8-D state, 5 Euler steps, 10-step horizon
of which 5 actions execute, 15 settle steps with the gripper open, 320
environment steps without early termination.

## Scene interchangeability (CPU-verified before launch)

The ten LIBERO-Goal BDDL files differ only in `:language`, `:obj_of_interest`
and `:goal`. LIBERO re-samples fixture placement on every hard reset; after
`env.seed(s)` the compiled MJCF is identical across all ten tasks (and differs
between seeds). All 500 stored init states (79-D) load exactly into every
task's environment, no goal holds at any init state or after settling, and
LIBERO's own `_check_success` equals the goal checker on every load. So any
goal can be instructed on any init state. Each episode uses env seed
`1000 + 100*source + trial`, shared by every condition on that init state.

## Conditions and budget

Frozen weights (fingerprint checked before and after), initial flow noise drawn
per (source task, trial, chunk) with base seed 2026, so all instructions on an
init state share their noise independent of batching.

| Condition | Episodes | Design |
| --- | --- | --- |
| Explicit | 120 | Each task's own instruction on its own init states, trials 0-11 |
| Vague | 40 | `complete the task`, `do the correct task` on trials 0-1 of every task |
| Swap | 180 | Trials 0-1 of every task under each of the 9 other instructions (all 90 ordered goal pairs) |

Total 340 episodes, 108,800 policy steps, in that phase order (explicit
trials 0-1, vague, swap, explicit 2-11) so a timeout still leaves every
condition represented. One GPU, 60-minute scheduler cap, at most two GPU jobs.
No training.

## Measurements

Every step records all ten goal predicates (LIBERO's `_eval_predicate` on each
task's parsed goal) plus LIBERO's native success for the env's own BDDL, which
must agree with the checker. Saved per episode: first step each goal held,
goals at the end, first-achieved goal(s), settled-state and MJCF hashes.

- Explicit success per task (instructed goal achieved at any step).
- Swap-follow: instructed goal achieved at any step (primary) and as the first
  goal achieved; per instructed goal, per ordered pair, full achieved matrix.
- Vague: per-goal achieved rate and first-goal distribution, per prompt.
- Gap = explicit rate - vague achieved rate, per goal.
- Goals reached when not instructed (candidate default behaviours).

## Provisional pass criteria (reported, not tuned)

Explicit >= 80% per goal; swap-follow >= 80% (overall and per goal); vague
per-goal achieved rate <= 1/k with k = 10 candidate goals; gap >= 30 points.
Per-goal swap counts are 18 episodes and pair counts 2, so pair-level results
are descriptive only.

## Launch

`outputs/libero-goal-headroom-thu2-20261029-r1/launch.sbatch` in a clean
worktree at the launch commit: storage probe, `gpu_probe.py` inside the step,
focused tests, CUDA model check on a real observation, then the run with a
45-minute driver cap. Raw `episodes.jsonl`, `summary.json`, receipts and logs
stay in the cluster output directory; small receipts and `RESULTS.md` are
mirrored here.
