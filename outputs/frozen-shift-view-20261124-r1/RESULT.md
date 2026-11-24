# Body view: result

**No native reward was produced.** This VM has no GPU or simulator. Whether a whole-episode Panda render rescues Kinova3 layouts 14, 15, 17, 36, 37, 38 is untested. Delivered: three adapters, a render twin, CPU fixture tests, archived-frame pixel counts and a launch script. [Plan](PLAN.md)

## Archived replay: pixels changed per camera

Ten archived 256 × 256 exposures from two Kinova3 command replays (layouts 14 and 15, controls 0–90), produced on CPU by the source-view run. Recount: `python analyze.py` → [frames.json](frames.json).

| Layout | Control | External changed | Wrist changed | Kinova re-render changed (ext/wrist) | Scene px newly covered by Panda | Scene px uncovered | Panda px hidden behind scene | Arm px in wrist (Kinova/Panda) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 0 | 2,432 | 0 | 0/0 | 923 | 286 | 680 | 0/0 |
| 14 | 30 | 6,811 | 0 | 0/0 | 1,905 | 795 | 57 | 0/0 |
| 14 | 45 | 6,357 | 0 | 0/0 | 1,977 | 720 | 90 | 0/0 |
| 14 | 60 | 6,939 | 0 | 0/0 | 2,010 | 599 | 49 | 0/0 |
| 14 | 90 | 6,094 | 0 | 0/0 | 1,706 | 537 | 85 | 0/0 |
| 15 | 0 | 2,301 | 0 | 0/0 | 793 | 275 | 695 | 0/0 |
| 15 | 30 | 6,850 | 0 | 0/0 | 2,178 | 549 | 29 | 0/0 |
| 15 | 45 | 8,010 | 0 | 0/0 | 3,155 | 982 | 301 | 0/0 |
| 15 | 60 | 8,539 | 0 | 0/0 | 3,469 | 952 | 453 | 0/0 |
| 15 | 90 | 10,263 | 0 | 0/0 | 4,209 | 1,558 | 259 | 0/0 |

- External camera: 2,301–10,263 of 65,536 pixels change (1,946–8,244 at the 224 × 224 policy input, from the prior readback). Wrist camera: 0 in every frame.
- Overdraw: the Panda covers 793–4,209 scene pixels the Kinova left visible and uncovers 275–1,558. The render is depth-tested, so 29–695 Panda pixels are correctly hidden behind foreground geometry; the prior run read back zero visible Panda pixels behind the scene and zero changes outside the two arm masks.
- **Whether those covered pixels are task objects is not settled.** The archive has no object-ID segmentation. A colour proxy on changed pixels that still showed reset-frame content gives table 0–578, wall 480–1,851, dark 167–757, other 27–423 per frame; "other" could contain drawer or bowl pixels and the proxy is not validated. The newly-covered count is the per-frame upper bound.

## Wrist treatment

Every arm in this benchmark carries the same PandaGripper with the wrist camera on it. With the Panda at the Kinova grip site, the archived wrist camera poses agree to 6.3e-16 m and 1.5e-15 rad, and neither arm has a pixel in any archived wrist frame (zero wrist pixels changed on the 117 Panda-render boundaries of the prior native run). The source wrist view at that pose is therefore the target wrist view. `view_panda_both` applies the same depth-tested substitution to the wrist image and nothing else: it equals `view_panda_external` except at replans where an arm link enters the wrist frustum, which the per-replan `inputs_changed` record and `views.jsonl` will count. No other wrist edit has support in the evidence.

## Adapters (`adapter.py`)

| Name | Policy input |
| --- | --- |
| `view_kinova` | Kinova arm pixels from an independent Kinova render, both cameras |
| `view_panda_external` | Panda at the grip site in `agentview`; wrist untouched |
| `view_panda_both` | Same, plus the same substitution in `robot0_eye_in_hand` |

Prompt, noise, candidate count, choice and command are untouched. Kinova3 starts only; a Panda start passes through unchanged and other arms are refused.

Images come from `twin.py`, a second simulator process that reuses `source_view.Exposure`, `Sibling` and `compose` unchanged with the validity checks of the prior native run. The adapter gives it the public reset observation (to identify the start among the 19 registered ones), the commands the simulator consumed, and each public observation; the twin must reproduce that observation bit for bit before rendering, else the episode faults. The twin's simulator state is privileged; it is not passed to the adapter, which receives only images and integer pixel counts.

## CPU fixture

`uv run --no-sync pytest -q -W error outputs/frozen-shift-view-20261124-r1`: 8 passed. Synthetic plant and policy, no MuJoCo. Covers: Kinova re-render is command-identical to the untreated branch; each Panda adapter changes only its cameras; the twin receives exactly the consumed commands, including an episode ending inside a window; a twin 1e-9 off the public trajectory faults; start identification and caching; Panda pass-through; panel discovery.

Not exercised anywhere: `twin.Scene` (MuJoCo backend) and the twin subprocess transport. They were checked by reading against `live.py`, `ur5e_sim_worker.py` and `env.sh` only.

## Native launch (not run)

```
bash outputs/frozen-shift-view-20261124-r1/run.sh "$(git rev-parse HEAD)" native01
```

Runs the shared panel (untreated, repeat, three adapters) on the 11 development starts; panel output lands in `outputs/frozen-vla-proposal-shift-20261124-r1/native01/`, render readbacks in `outputs/frozen-shift-view-20261124-r1/native01/views.jsonl`.

## Open

- First native use should be a one-start smoke: `run.sh <pin> native00 --starts 14`. Likely failure points: bitwise agreement of twin and harness observations across processes, and three extra MuJoCo contexts on the GPU.
- A new start costs up to one twin reset per earlier unseen registered start (none in panel order).
- Object-ID overdraw counts need a segmentation pass in the native trace; not implemented.
