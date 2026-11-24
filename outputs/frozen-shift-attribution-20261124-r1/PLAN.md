# Attribution lane: which input moves frozen proposals toward a tape that works on Kinova3?

Written November 24 before any result of this lane. [Program plan](../frozen-vla-proposal-shift-20261124-r1/PLAN.md). No adapter; `adapter.py` exposes an empty `ADAPTERS`.

## Question

On Kinova3 states visited by an IIWA command tape that earned native reward on Kinova3, which change to the frozen π₀.₅ conditioning (camera body, proprioceptive state, prompt) moves its next five proposed controls toward the tape's next five controls, beyond what a fresh policy-noise draw does?

## Prediction

Raw Kinova3 proposals sit farther from the donor's next five controls than the donor arm's own proposal does (which is the tape, distance near zero by construction). If the failure is a proposal failure caused by observing Kinova3, at least one swap (Panda render, mapped state, a subtask prompt) reduces that distance at the same noise, and by more than fresh noise changes it.

## What counts against it

- No swap is closer to the donor than raw at the same noise, or every swap's change is inside the spread of the fresh-noise control. Then the measured input channels do not carry the shift, or distance to one donor tape is the wrong yardstick (stated either way).
- Raw Kinova3 proposals are already as close to the donor as fresh-noise spread allows. Then these states are not where the proposals depart.
- The tape does not earn reward when replayed on the GPU host. Boundaries after that point are scored separately; the CPU donor success does not carry over.

No numeric gate. The comparison that decides is swap versus fresh noise, paired on the same boundary, with intervals over layouts.

## Prior work checked

- [Donor](../frozen-transfer-donor-20261124-r1/RESULT.md), [validation](../frozen-transfer-donor-validation-20261124-r1/RESULT.md): IIWA tape 35 earned reward on Kinova3 34, 35, 37 and tape 34 on 14, on Baige CPU with blank rendering. Tapes store per-boundary `first_five`, `sent`, noise digest. GPU and CPU settled physics hashes differed on every start.
- [Source view](../frozen-vla-source-view-20261123-r1/NATIVE.md): Panda render changed all ten proposals on Kinova3 14/15 frames; full 10×7 proposals archived. Distances were to raw, not to a successful tape.
- [State transfer](../frozen-vla-factory-20261123-r1/state-transfer/RESULT.md) scored state adapters against a known-TCP output, not against a successful target tape.
- Selector prompt check: SmolVLA only, one Panda frame.
None of these scored proposals against a donor tape at donor-visited Kinova3 states.

## Protocol

Pairs (target Kinova3 layout ← IIWA tape): 34←35, 35←35, 37←35 (seed 42710300+layout), 14←34 (seed 41810314, as validated). Replay `sent` through `sim.call('chunk')` five controls at a time with real rendering. Before each window, query the frozen policy. Noise for every non-control condition is the donor's own draw for that boundary (bank seeded donor seed + 817117, digest checked against the tape).

| Condition | Change |
| --- | --- |
| `raw` | none |
| `raw_repeat` | none, boundary 0 only (determinism) |
| `noise_a`, `noise_b` | two fresh noise draws, raw inputs (control) |
| `panda_external` | Panda render in the external camera |
| `panda_both` | Panda render in both cameras |
| `state_mapped` | state packed in the source convention (`canonicalize_quat`, the motor's existing public rule); other lanes' `Adapter.observation` can be added by name |
| `state_zero` | state vector zeroed |
| `state_shuffled` | fixed derangement of the eight state coordinates |
| `prompt_open`, `prompt_put` | "open the top drawer", "put the bowl in the top drawer" |

Score, on the first five controls clipped to [−1, 1]: RMS distance to the donor's next controls (`sent`), to the donor arm's own proposal (`first_five`), and to `raw`; also per channel group (xyz, rotation, gripper). Contrasts per boundary: condition − raw and condition − mean fresh noise. Report mean per layout and over layouts, with the interval from all ordered resamples of whole layouts (4 layouts, 256 resamples); boundaries inside a layout are never resampled separately. Report all boundaries and boundaries up to the replay's first reward. Native reward of the replay is audit data, recorded, not used by any condition.

Conditions use public observations, the donor tape and noise only.

## Evidence classes

- CPU fixture: `test_attribution.py`, synthetic simulator and policy.
- Archived (CPU, now): `archive.py` scores proposals already in the repository against the donor tapes. Boundary 0 is teacher-forced-equivalent (reset state); later archived frames are Kinova3's own closed-loop states and are labelled off-tape.
- Offline forward-pass (GPU job, not run here): `run.sh`. 146 boundaries × 10 queries + 4 repeats = 1,464 policy calls. At the recorded 0.18 s per replan including simulation (IIWA 35 raw: 6.6 s for 37 replans) this is about 5 minutes of inference; with model load, JIT and Panda sibling renders the estimate is 0.25 GPU-hours on one RTX 4090, capped at 1 hour.

Nothing here is task success. No fresh native reward is produced on this VM.
