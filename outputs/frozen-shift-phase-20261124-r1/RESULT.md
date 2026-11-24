# Phase prompt: result

**No native rollout ran, so there is no task-success evidence for this lane.** The adapters, a CPU readback of the archived Kinova3 failures and a Baige launcher are delivered. [Plan](PLAN.md)

## Evidence by kind

| Kind | What exists |
|---|---|
| Native reward | None. |
| Offline forward pass | None. pi0.5 was not called. |
| Archived frames and tapes | [schedule.json](schedule.json) from [schedule.py](schedule.py): reader on untreated Kinova3 failure tapes. |
| CPU fixture | 13 tests in [test_phase.py](test_phase.py) on the shared harness fixture. Its rewards are synthetic and are not asserted on. |

## Adapters ([adapter.py](adapter.py))

| Name | Prompt at each replan |
|---|---|
| `phase-reader` | "open the top drawer" until the camera reader's source majority says open, then "put the bowl in the top drawer". Returns to opening only when a visible drawer front is below every source crossing (18.90–19.59 px). Holds inside that band and while the front or the reset front is hidden. |
| `phase-instant` | `Signals.cue` applied every replan: no hold, a hidden front reads closed. Isolates the hysteresis. |
| `phase-clock` | Opening until control 60, then placement (source-fitted clock from the phase-transfer model). |
| `phase-open`, `phase-place` | One prompt for the whole episode. |

The original prompt is the panel's `raw` and `repeat` branches. Only the prompt changes: images, state, noise, candidate count and commands are untouched (fixture check on every adapter). Inputs are the public agentview image and the count of issued controls.

## When each prompt would be active (archived, untreated trajectories)

These are schedules along the **original-prompt** failure trajectories. A held prompt changes the trajectory from control 0, so they are not the adapter's native schedule.

| Layout | Evidence | Opening prompt | Placement prompt | Clock |
|---|---|---|---|---|
| 14 | Current reader on five archived frames (0, 30, 45, 60, 90) | 0–45 read closed (0.18 px at 45) | 60 and 90 read open (20.3, 21.2 px) | switches at 60 |
| 15 | Current reader on five archived frames | all five read closed (0.0 px) | never | switches at 60 |
| 36 | Previous four-tile reader scores in the cue tapes, raw / tuned | 0–95 / 0–60, again from 245 / 185 | 95–245 / 60–185 | switches at 60 |
| 37 | Previous four-tile reader scores, raw / tuned | 0–60 / 0–60, raw again from 170 | 60–170 / 60–300 | switches at 60 |
| 17, 38 | No frame and no reader score is archived | — | — | switches at 60 |

- **14 and 15, audit.** The reader agrees with the simulator drawer joint (CPU replay of the same tapes, audit only) on all ten frames. That joint first passes the native open bound at replan 60 on layout 14 and stays open to control 300; on layout 15 the drawer never moves. If the reader keeps agreeing after control 90, layout 14 gets opening 0–60 then placement 60–300, the same as the clock, and layout 15 gets the opening prompt for all 300 controls while the clock asks for placement into a closed drawer from control 60.
- **36 and 37.** The current reader was not run: these tapes hold scores and image digests, no images. The late returns to "closed" (36 raw at 245, 36 tuned at 185, 37 raw at 170) cannot be told apart from an arm or bowl covering the strip. `phase-reader` and `phase-instant` differ in exactly that case.
- **Reader coverage.** Frames exist only up to control 90 on two layouts. Reader behavior with the arm over the drawer late in an episode is untested on Kinova3.

## What the native run decides

The panel compares each adapter with `raw` and `repeat` on the eleven development starts (failures 14, 15, 17, 36, 37, 38 and 16, 31, 34, 35, 39), paired by layout. `phase-reader` against `phase-clock` is the registered contrast; `phase-open` and `phase-place` show whether either prompt alone accounts for a change; `phase-instant` shows whether the hold matters. On the archived evidence the reader and clock schedules coincide on layout 14 and differ on layout 15.

## Launch

In a Baige Distributed Training allocation with one RTX 4090, from a clean checkout of the pinned commit:

```
bash outputs/frozen-shift-phase-20261124-r1/run.sh <head sha of this branch> phase01
```

It calls the shared [run.sh](../frozen-vla-proposal-shift-20261124-r1/run.sh) with `--starts development` and the five adapters (77 episodes of at most 300 controls), and writes per-replan drawer-front pixels and scores to `phase-trace/` next to the panel records. The launcher was syntax-checked and its start and adapter resolution run with `panel.py --list` here; the script itself has not run on Baige.

## Not done

- Native rollouts and any offline pi0.5 forward pass.
- Reader schedules for layouts 17 and 38, and current-reader schedules for 36 and 37 (no archived images).
- `docs/FINDINGS.md` and the coordination board were left alone as instructed.

## Reproduce

```
uv run --no-sync pytest -q -W error outputs/frozen-shift-phase-20261124-r1
uv run --no-sync python outputs/frozen-shift-phase-20261124-r1/schedule.py
```
