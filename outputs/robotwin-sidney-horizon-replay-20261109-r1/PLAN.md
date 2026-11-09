# Sidney phone stand: 400 versus 1,200 action targets

## Question

Sidney's card reports 66/100 on clean `place_phone_stand`. The run that
produced it used `lerobot-eval --env.type=robotwin`. Our native panel
(`robotwin-sidney-native-sft-20261109-r1`, job `job-5tj6l8qni06f`) was at
13/63 on 2026-11-09 17:36 Shanghai. The two protocols differ in four ways:

| | LeRobot wrapper (`92f96f33`) | Native panel |
| --- | --- | --- |
| Action limit | No `eval_mode`, so `_eval_step_limit.yml` is never read; LeRobot stops at `episode_length=1200` | 400 targets |
| Prompt | `place phone stand` | Scene-bound unseen instruction |
| Seeds | `lerobot-eval` `seed=1000`, so 1000–1099 | 100000 onward |
| Unsolvable seeds | Counted as failures | Expert-solvable only |

Cameras (D435, 320×240), H50/K10 and executing all 50 actions per chunk
match. This run measures how much of the gap the action limit alone explains.

## Design

Rerun the same 100 clean cases in `PHONE100-CASES.json` with the same seeds,
prompts, model bundle, worker, deterministic runtime and per-chunk noise seeds
(`1000 + 100*index + chunk`). The only change: after `setup_demo` asserts the
native limit of 400, set `task.step_lim = 1200`.

Capping actions only stops an episode, so each 1,200-target episode contains
its 400-target episode as a prefix. Report from the same episodes:

- success by 400 targets (`success and targets <= 400`);
- success by 1,200 targets;
- the per-case pair, with an exact McNemar test (only 400-fail/1,200-success
  pairs are possible).

Cross-check determinism against `job-5tj6l8qni06f` once it is terminal:
per case, compare success by 400 and the action SHA256 of every chunk the
original episode delivered. Agreement makes this a counterfactual on the
original panel. If they disagree, the within-run 400/1,200 comparison still
holds, and the mismatch rate is recorded.

Clean domain only. The card publishes no randomized phone-stand number, so a
randomized horizon run would have no reference.

## Interpretation

- Success by 1,200 near 60% or higher: the action limit accounts for most of
  the gap. Then decide which limit to report and use for PPO/GRPO; the native
  400 remains RoboTwin's official limit.
- Still near 20–30%: the limit is not the cause. Next is stage 2 below.

## Stage 2 (not launched; decided after stage 1)

Reproduce the card's protocol directly with LeRobot's own `RoboTwinEnv`:
seeds 1000–1099 with no expert filter, prompt `place phone stand`, 1,200
steps, `eval_mode` unset. Then change one factor at a time toward the native
panel (prompt, seed range).

## Execution

- One Baige Pro6K GPU, 420-minute ceiling (25,200 GPU-s = 7.0 GPU-h), no
  automatic retry. Within the epoch-2 budget of 100 GPU-h (2.88 closed before
  this run, plus 6.17 reserved by `job-5tj6l8qni06f`).
- Source: the `main` commit that adds this directory; record it in
  `LAUNCH.json`. Launch with `launch.sh <commit> 100 01`.
- GPU probe in the job step; the first episode exercises the full path and is
  audited before monitoring continues. Check progress every 20 minutes.
- Raw outputs: `/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/sidney-horizon-replay-20261109-r1/<id>`.
  Tracked: `LAUNCH.json`, `LIVE-CHECK.json`, `RESULT.md`, readback.
- Does not touch `job-5tj6l8qni06f`, its outputs, or its coordination row.
