# Does a held, camera-switched subtask prompt rescue the Kinova3 drawer failures?

Lane plan, committed before any lane result. Program: [PLAN.md](../frozen-vla-proposal-shift-20261124-r1/PLAN.md). Harness: `devin/1795493348-frozen-shift-harness` at `5fd6c426`. Frozen π₀.₅ weights and motor parameters never change. Only the prompt string given to the policy changes.

## Question

Frozen π₀.₅ times out on Kinova3 layouts 14, 15, 17, 36, 37, 38 under the composite instruction "open the top layer of the drawer and put the bowl inside". Does conditioning every replan on the current subtask, chosen from the public camera, move it onto a behaviour that finishes: "open the top drawer" while the drawer reads closed, "put the bowl in the top drawer" while it reads open, and back again if it recloses?

## What a held prompt changes relative to the one-time cue

The one-time cue is negative: [camera-timed cue](../frozen-transfer-cue-20261124-r1/NATIVE.md) rescued 0/2 Kinova3 failures (36, 37) and lost one success; the [selector](../selector-native-launch-20261124-r1/RESULT.md) cue branch rescued 0/3 (36–38). Both changed the text for **one** replan (five controls) and restored the composite instruction at the next query. Three things differ here:

1. **Opening phase.** The cue left the first 45–60 controls on the composite instruction. Here "open the top drawer" conditions the policy from control 0, so the whole pre-opening trajectory is a different sample path. The cue could not test whether the failure is set up before the drawer opens.
2. **Persistence.** After opening, every replan (about 48 of 60) carries the placement prompt, not one. A single five-control perturbation followed by the original conditioning can be absorbed; a held prompt cannot revert.
3. **Reclosure.** The cue fires once. In the archived cue tapes the previous reader's open score returned to near zero late on 36 (both raw and tuned) and on 37 raw, and two of eight earlier replays reclosed the drawer. A one-time cue has no answer to that; this adapter returns to the opening prompt.

Known risk, stated before the run: neither subtask string is a trained LIBERO instruction for this checkpoint, and the earlier phase study warned that a persistent placement instruction can outlive the open drawer. Hysteresis (below) addresses the second; the first is what the native run measures.

## Prediction

If the hypothesis is right, `phase-reader` rescues at least one of the six failures that `raw` and `repeat` both fail, and does so on a start where `phase-clock` does not, without losing more of the known successes (16, 31, 34, 35, 39) than the untreated repeat's own discordance.

## What counts against it

- `phase-reader` equals `phase-clock` on the six failures (the program's stated criterion): then camera grounding of the phase adds nothing over elapsed time.
- `phase-reader` rescues nothing beyond the repeat's discordance: then held subtask conditioning does not move the policy off the failing behaviour.
- Rescues matched by equal losses on the known successes.
- If the constant prompts (`phase-open`, `phase-place`) match `phase-reader`, the switch itself is not the active ingredient.

No numeric threshold is set. Decisions follow the paired counts and layout bootstrap intervals the panel reports against `raw` and `repeat`.

## Prior work checked

- [Phase transfer](../frozen-vla-phase-transfer-20261123-r1/RESULT.md): four-tile reader and source clock (`evidence/model.json`); fixed-command replays only, no closed-loop prompt.
- [Camera-timed cue](../frozen-transfer-cue-20261124-r1/NATIVE.md), [selector native](../selector-native-launch-20261124-r1/RESULT.md): one-replan cue, negative.
- [Current drawer reader](../frozen-vla-selector-20261124-r1/drawer-evidence.md): `signals.py` `drawer_front`, `drawer_probability`, `Signals.cue`, frozen on Panda (`drawer-calibration.json`); 0 premature and 0 missed on 488 archived Kinova3 frames from layouts 30–33; no native reward test of a held prompt exists.
- No earlier run holds a subtask prompt for a whole episode on π₀.₅.

## Adapters (`adapter.py`, `ADAPTERS`)

All use only `prompt`, `reset` and `observe`. Images, state, noise, choice and command stay untreated. Inputs are `agentview_image` and the count of executed controls.

| Name | Prompt at each replan |
| --- | --- |
| `phase-reader` | Hysteretic camera phase: opening prompt until the reader reports open, placement prompt while open, opening prompt again after a reclosure |
| `phase-instant` | `Signals.cue` rule each replan with no memory (score > 0.5, unobservable edge reads closed). Isolates the hysteresis |
| `phase-clock` | Opening prompt until the source-fitted clock in the phase-transfer `model.json` first reaches 0.5 (control 60), then placement. Never returns |
| `phase-open` | Opening prompt for the whole episode |
| `phase-place` | Placement prompt for the whole episode |

The original prompt is the panel's own `raw` and `repeat` branches.

Hysteresis, derived from the reader's own calibration and not from a tuned number. The score is the fraction of 18 Panda source crossings that the current drawer-front displacement exceeds. Switch to placement on a strict majority (score > 0.5, the `cue` rule). Return to opening only when the edge is observed and the displacement is below every source crossing (score 0). Inside the source-crossing band, and whenever the edge or the reset edge is unobservable (arm or bowl over the strip), keep the current prompt. `Signals.cue` instead reads an unobservable edge as closed, which under a held prompt would send "open the top drawer" while the arm carries the bowl over an open drawer.

## Protocol

1. CPU fixture: `test_phase.py` against the harness fixture. Each adapter changes only the prompt record; schedule, hysteresis, reclosure, unobservable edge, clock time and per-episode reset are checked. Synthetic rewards, not task evidence.
2. CPU archived readback (`schedule.py` → `schedule.json`): run the reader and the adapter state machine over every archived Kinova3 failure frame in the repository for layouts 14, 15, 17, 36, 37, 38 and report when each prompt would be active. Stated in advance: full images exist only for layouts 14 and 15 (five frames each, [source view](../frozen-vla-source-view-20261123-r1/evidence/cpu01)); layouts 36 and 37 have only the previous four-tile reader's per-replan scores in the cue tapes; 17 and 38 have neither. Audit drawer joint travel for 14 and 15 comes from the [reachability replay](../frozen-vla-reachability-20261123-r1/evidence). These are schedules along **untreated** trajectories. They do not say what the policy does under the held prompt.
3. Native (not run here, no GPU or Baige on this VM): `run.sh <pin> <attempt>` calls the harness launcher with this lane's adapters on the development starts, one RTX 4090, Baige Distributed Training. 11 starts × (raw, repeat, 5 adapters) = 77 episodes, about 0.3 GPU-hours at 0.0037 GPU-hours per episode. A rescuing adapter is then locked for layouts 40–47 and Panda 0–3 per the program plan.

Evidence delivered by this lane on this VM: CPU fixture and archived readback only. No native reward.
