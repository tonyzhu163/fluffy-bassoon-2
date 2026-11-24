# Teacher-forced attribution: code qualified on CPU, GPU forward pass not yet run

No native reward and no fresh forward pass were produced here. The only measured result is a CPU rescoring of ten archived Panda-render proposals.

## Archived replay (GPU proposals of `job-69hluhi5wgb7`, rescored on CPU)

Ten Kinova3 frames, layouts 14 and 15 at controls 0, 30, 45, 60, 90, raw versus Panda in the external camera at equal noise. Distance is RMS over the first five clipped controls to the IIWA tape at the same control index.

| Donor tape | Raw distance | Panda − raw | Range over layout resamples | Layout 14 / 15 | Frames closer |
|---|---:|---:|---|---|---:|
| IIWA 34 (rewarded on Kinova3 14) | 0.1808 | −0.0045 | −0.0067 to −0.0023 | −0.0022 / −0.0069 | 4/10 |
| IIWA 35 | 0.2061 | −0.0077 | −0.0087 to −0.0068 | −0.0088 / −0.0067 | 5/10 |

- Panda moved each proposal by 0.009–0.083 RMS, but toward the donor by only 2–4% of the raw distance, and on half the frames or fewer.
- Limits: two layouts (four ordered resamples, so the range is descriptive); only control 0 is a state the donor replay visits (there: −0.0005 and −0.0077, 1/2 frames each); later frames are on Kinova3's own closed-loop path; noise is the Kinova3 episode's, not the donor's; no fresh-noise draw is archived, so no noise floor.
- Raw repeat and Kinova render equal raw exactly. The donor's issued controls are its own clipped proposal (max difference 0.007), so the two targets nearly coincide.
- No archived proposals exist for the state or prompt conditions on pi0.5. [Per-frame and per-group scores](evidence/archived.json) · [Scorer](archive.py)

## CPU fixture

`uv run --no-sync pytest -q -W error outputs/frozen-shift-attribution-20261124-r1`: 10 passed. Synthetic plant and policy; rewards there are not evidence. Checks: the donor tape is the only thing executed; each condition changes only its own policy input; donor noise is reused and the two fresh draws differ; missing Panda renders are recorded, not imputed; lane adapters see public observations only; intervals resample whole layouts; the registered tapes verify against their manifest (146 boundaries, 1,464 calls).

## Not done

- The forward-only job. `bash outputs/frozen-shift-attribution-20261124-r1/run.sh $(git rev-parse HEAD) gpu01` in a one-RTX-4090 Distributed Training allocation: 4 replays, up to 723 controls, 146 Panda renders, 1,464 policy calls, estimated 0.25 GPU-hours, one-hour wall limit. It writes `teacher-<layout>-from-<tape>.json` and `summary.json` (per condition: distance to donor controls, to the donor arm's proposal and to raw; paired minus raw and minus the mean of two fresh-noise draws; intervals over layouts; a separate scope for replays that earned reward).
- `views.py` (Panda render attached to each boundary, adapted from the source-view `live.py`) has never executed: it needs the pinned simulator. `--views none` drops the two render conditions if it fails.
- The archive found zero changed wrist pixels under Panda, so `panda_both` may equal `panda_external`; the input digests in each record will show it.
- `state_mapped` is the motor's `canonicalize_quat` rule. Other lanes' state maps can be scored with `--adapters <name>`.
- The donor replays earned reward on Baige CPU; on the GPU host the replay may not (physics hashes differed between hosts). `summary.json` reports replay reward per layout as audit data.
