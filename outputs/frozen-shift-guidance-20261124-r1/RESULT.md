# Source guidance: result

Status: **CPU-qualified, native run not launched** (no GPU or Baige access on this VM). No task-success claim
is made. Plan: [PLAN.md](PLAN.md). Numbers: [analysis.json](analysis.json) from `analyze.py`.

## Evidence kinds

| Evidence | Kind | What it shows |
|---|---|---|
| `analyze.py` on six archived source successes | offline, archived tapes | alignment quality only |
| Archived Kinova3 raw tapes 34-37 against donors | offline, archived tapes | diagnostic only |
| `test_guidance.py` (11 tests) | CPU fixture, synthetic rewards | the code does what the plan says |
| Fresh native reward | **none** | adapter effect on the task is unmeasured |

## Offline: does progress alignment predict held-out source successes?

Six source successes (IIWA 34, 35, 36, 37; Panda 14, 15), each held out in turn and predicted from the five on
other layouts. Error = norm of (donor window's summed five-control translation − the held-out tape's), normalized
action units, averaged over the held-out tape's windows and its donors.

| Rule for picking the donor window | Mean error, six tapes (range) |
|---|---|
| Same phase, nearest within-phase progress (**this lane**) | 1.68 (1.31-2.24) |
| Same control index (time-aligned) | 2.87 (2.18-3.90) |
| Mean over windows of the same phase | 4.03 (3.86-4.12) |
| Mean over all donor windows | 4.85 (4.79-4.90) |

Paired, time minus progress: +1.20, bootstrap over the six tapes 95% [+0.91, +1.46], positive on 6 of 6.
Donors from the other arm only: +0.98 [+0.61, +1.38], 6 of 6. Every query phase was present in every donor.
Six tapes from two arms and one instruction: the interval is over tapes, not over layouts in general.

- **Tool frame.** Panda 14↔15 only (the IIWA tapes archived no orientation): expressing displacement in the
  frame turned by the tool's rotation since reset gave 1.25 and 1.69 against 1.09 and 1.51 in the reset frame
  (tool turned at most 16° and 25°). Two tapes; no support for the rotating frame, so native adapters use the
  reset frame. `Guidance(..., frame="tool")` is implemented and tested but not registered.
- **Jaw.** On the Panda tapes the aperture never widened in a window commanded closed (0 of 31) and narrowed twice
  in 49 commanded open, so the sign of the issued jaw command stands in for aperture on donors without it.
- **Kinova3 diagnostic.** Mean gap from each executed window to its nearest progress-aligned donor window:
  successes 34 and 35: 0.84, 0.91; timeouts 36 and 37: 2.05, 2.07. Both timeouts opened the drawer, never
  commanded the jaw closed, and the camera reader returned to "shut". Two against two, untreated tapes: it says
  failed episodes leave the source corridor, not that steering back fixes them.

## What is delivered

- `donors.py`: digest-checked loaders, phase key (drawer open, jaw commanded closed) + summed issued translation.
- `adapter.py`: `ADAPTERS = [guidance_iiwa3435, guidance_random]` for the shared panel (donors on layouts 34 and
  35, disjoint from the six failures and locked 40-47). `for_layout` adds `guidance` (all other-layout donors),
  `guidance_time`, `guidance_same` (upper reference; exists for 14, 15, 34-37).
- `run_guidance.py`, `run.sh`: lane driver over the shared `panel.run_start`, so branches share reset, opening
  proposal and the raw/repeat pairing; writes per-replan choices. `run.sh` is the shared `run.sh` with the entry
  point swapped.

Native launch, inside a Baige Distributed Training job with one RTX 4090, at this branch's head:

    bash outputs/frozen-shift-guidance-20261124-r1/run.sh "$(git rev-parse HEAD)" native01 --starts development

Shared-panel form: `bash outputs/frozen-vla-proposal-shift-20261124-r1/run.sh "$(git rev-parse HEAD)" <attempt>
--starts development --adapters guidance_iiwa3435,guidance_random`.

## Deviations from the brief, and limits

- Donor drawer state: IIWA tapes carry the archived camera gate (public). Panda tapes have no camera frames, so
  their drawer flag is the archived drawer joint at the −0.14 m threshold the reader is calibrated to. Donor-side
  only; the online query uses the camera reader.
- Donor aperture: absent on IIWA tapes; phase uses issued jaw sign on every tape and online (audit above).
- "Tool frame" is rotation since reset, which needs no shared tool-frame convention across arms. It does not
  cancel a differently oriented base; LIBERO arms share base orientation, and commanded displacements are
  already free of base position.
- The shared harness passes no layout to `reset`, so the layout-aware variants need `run_guidance.py`.
- The harness's `reset` reply includes `arm`, `joint_count`, `max_steps`; this lane reads only the first
  observation.
- Not measured: whether four samples differ enough at a replan for any choice to matter (no archive holds
  several proposals per replan); the camera reader on Kinova3 with K=4 online; anything native.
- `run.sh` and the real-policy path of `run_guidance.py` were syntax-checked and exercised on the fixture only.

## Distinctions

- Embodiment-Classifier Guidance scores samples with target-arm successes; this lane uses none.
- The demonstration handoff replays donor commands and then releases the motor; here every executed control is
  one of the frozen policy's own samples and no donor command is ever sent.
