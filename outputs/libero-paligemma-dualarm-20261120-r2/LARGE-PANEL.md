# Generated Goal 5 BASE source qualification

Development uses 50 distinct layouts paired across Panda and Sawyer. Update 1600 from the existing `sft01/fit` is the first candidate. Each arm contributes 50 fresh closed-loop episodes. The separate sealed panel has 100 new layouts and 100 episodes per arm. The full manifests and outcome tapes live on project NVMe.

## Development CLI

From the repository root on allocated Baige, set `PIN` to the deployed revision. Each invocation of the existing launcher needs a fresh attempt name. All stages share the same persistent `panel` directory.

```bash
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
fit=$campaign/sft01/fit
panel=$campaign/large-panel-v1
bash "$run/run.sh" "$PIN" lpdevprep01 sim large_panel.py prepare --root "$panel" --phase dev --fit "$fit"
bash "$run/run.sh" "$PIN" lpdevscore01 vlm large_panel.py source --root "$panel" --phase dev --fit "$fit" --step 1600
bash "$run/run.sh" "$PIN" lpdevsum01 sim large_panel.py aggregate --root "$panel" --phase dev --fit "$fit" --step 1600
```

The prepare command publishes `dev/manifest.json` and `dev/rows/*.npy` before scoring. Source writes one native tape and digest per layout and arm beneath `dev/scores/<checkpoint-sha>/stride-5/`. Aggregate writes `report.json` there only after all 100 episodes validate. The default rollout uses eight independent native processes, one batched frozen motor, and five issued controls per replan. One arm or a zero-based layout shard can be selected with `--arm Panda` or `--shard-index I --shard-count N`.

## Development selection and sealed test

Run the same `source` and `aggregate` commands for each registered 6400-update fit, changing `--fit`, `--step` and the launcher attempt names. Keep `--root` fixed. The three-candidate comparison uses stride 5. `accept` rereads every development tape, ranks by the smaller arm count, then the combined count, then registration order. The winning checkpoint needs at least 38/50 successes on **each** arm. Set `CONVENTIONAL_FIT` and `SERVO_FIT` to the two extended fit directories. [Registered decision](PLAN.md)

```bash
bash "$run/run.sh" "$PIN" lpaccept01 sim large_panel.py accept --root "$panel" --fit "$fit" \
    --extended-fits "$CONVENTIONAL_FIT" "$SERVO_FIT"
```

Acceptance writes `dev/comparison.json`, `accepted/checkpoint.pt` and `accepted/selection.json`. Use the winning fit directory and step from the comparison for all test commands. The evaluator loads the saved checkpoint bytes. Test preparation verifies selection before sampling seed 840000.

```bash
bash "$run/run.sh" "$PIN" lptestprep01 sim large_panel.py prepare --root "$panel" --phase test --fit "$WINNER_FIT"
bash "$run/run.sh" "$PIN" lptestscore01 vlm large_panel.py source --root "$panel" --phase test \
    --fit "$WINNER_FIT" --step "$WINNER_STEP"
bash "$run/run.sh" "$PIN" lptestsum01 sim large_panel.py aggregate --root "$panel" --phase test \
    --fit "$WINNER_FIT" --step "$WINNER_STEP"
```

Only complete 100/100 or 200/200 episode panels produce a report. A completed evaluation writes its counts and returns zero even when a gate fails. Read `passed` for the gate decision. The test requires at least 75/100 separately on Panda and Sawyer before a target reset. The Wilson intervals describe uncertainty for each arm separately. Each episode lists the native seed, layout ID, scene and movable-placement fingerprints, native return and tape SHA-256.

## Layouts and native controls

Development reserves seeds 740000–749999 and test reserves 840000–849999. Preparation checks all 50 released rows and their four-object XY configurations, the supplied fit's episode seeds and generated geometry, earlier selected layouts, and the entire development panel before accepting a test layout. Both source arms must reproduce the same pre-settle scene. After the official ten settling controls, every named task body's pose must agree within the native tolerances of 1 mm and 0.01 rad. The manifest saves the geometry, row arrays, source reset fingerprints and paired errors. A duplicate hash can advance the seed cursor. A sampler, collision, IK, native reset or pairing error interrupts preparation at that seed for diagnosis. Policy outcomes play no role in admission.

The model receives external RGB, wrist RGB, the eight public wrist/jaw values and the task instruction. The baseline, spatial-bin servo and waypoint loaders verify the original general PaliGemma MIX snapshot and frozen delta. The native interpreter receives a bounded action prefix, steps 20 Hz OSC, and returns the latest public observation plus measured motion and sparse reward for the tape. A zero-reward episode must reach all 300 controls. Each worker owns its simulator process. The evaluator fixes the model batch shape at eight even on a short shard. Use `--workers 1` through `8` to tune simulator concurrency. Retry interrupted scoring with the same arguments and a fresh launcher attempt name.

## Physical session bridge

On the qualified runtime, set `CROSSARM_SIM_PYTHON` as in the R2 launcher. The context manager starts a dedicated native interpreter. Pass a manifest `layout_id` to `reset`. The returned observation contains only the four public fields. `reset_info` holds the physical audit receipt separately.

```python
from large_panel_sim import Session

with Session(panel_root, "test", "UR5e") as env:
    obs = env.reset(layout_id)
    obs, measured_motion, native_reward, done, success = env.step(controller(obs))
```

Target initialization requires a fully audited passing sealed source panel. A target reset reloads the exact saved Panda donor array, checks the pre-settle scene fingerprint and compares the settled body poses with the paired Panda reference. A target that cannot reproduce the recorded layout raises an error on that ID. The caller can repeat a layout for paired conditions. [Native bridge](large_panel_sim.py) · [Evidence checks](large_panel_protocol.py)
