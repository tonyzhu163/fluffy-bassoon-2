# PR7 checkpoint reproduction and paired prediction export

The exporter is ready for the parent’s single-GPU Baige Distributed Training launch. Linux CPU qualification passed on November 23 with JAX 0.5.3 and OpenCV 5.1.0. The three refits and their numerical comparison require the scheduled training attempt. [CPU preflight](export-preflight.log)

## Fit and inputs

The question is whether the three tuned readers recover the published offline result before their predictions are used in native retries. A changed seed effect, a mismatch in tape identity, or a changed prediction after checkpoint restore would undermine that interpretation.

The launcher selects only PR7’s `perceiver_cont_warm` entry and calls the original `load_recorded`, `take_chunks`, and `train_model`. Training concatenates the eight `gpu12scale` shards in their original order, then takes the first three chunks of the 1,466 Panda/Sawyer traces. Each seed uses its original JAX initialization and NumPy minibatch and continuous-interface draws. The factor head uses a 24-position Perceiver, batch size 32, 2,500 updates, peak rate 0.0003 and 200 warmup updates. The other 21 positions remain in the saved parameter tree. All three SymMeta readers are the original same-seed Stage 1 checkpoints. [PR7 implementation](../symmeta-wider-family-20261123-r1/eval_wider_family.py) · [Input hashes from the original PR7 preflight](export-inputs.json)

The exporter checks each source shard against the original NPZ digest and collection-plan order. Each of the 384 physical cases must agree with the test plan and archived evidence on panel, index, suite, task, transform, passive path, support physics and tape bytes. On the locked 96 UR5e × π₀.₅ cases the decoded arrays, masks, labels, regime order and DIS maps must also agree with PR7’s original loader. The other panels contain 24 slots with valid chunks at indices 0–2. Tuned inference crops these to the same three inputs as PR7. Paper inference uses Stage 2A’s per-tape DIS and full temporal shape, three slots on the locked panel and 24 on the others. Both readers decode their own yaw, gain and gripper sign. [Locked tape evidence](../symmeta-camera-command-20261123-r1/evidence/roll.json.gz) · [Other panel identities](../symmeta-fk-crossarm-20261123-r1/evidence/)

## Offline training sanity

| Reader seed | PR7 last minibatch loss, tuned | PR7 OOD yaw MAE, tuned | PR7 OOD yaw MAE, paper | Baige CPU native OOD yaw MAE, paper |
|---|---:|---:|---:|---:|
| 41 | 0.0240 | 5.423745° | 6.823566° | 6.829198° |
| 42 | 0.0193 | 5.832224° | 7.000050° | 7.000029° |
| 43 | 0.0197 | 5.478292° | 7.287281° | 7.297013° |

The denominators are 64 OOD and 96 overall per seed. CPU preflight matched the 384 physical keys, the original 11 training and checkpoint hashes, all three original readers’ 288 archived gripper predictions and the first paper prediction on each additional panel to its stored CPU value. The latter three full-precision triples appear in the log. A newly initialized 24-position template saved and reloaded through the campaign reader with identical decoded factors on an actual cropped IIWA tape. CPU preflight consumed zero GPU-hours and performed zero optimizer updates. [Preflight output](export-preflight.log) · [Verified PR7 scores](../symmeta-wider-family-20261123-r1/gpu01w454/wider-family-summary.json) · [PR7 training log](../symmeta-wider-family-20261123-r1/gpu01w454/run.log)

## Launch and handoff

Use one exclusive RTX 4090 to match the original hardware if available. The launcher also recognizes the qualified Baige RTX PRO 6000D. Set the scheduler wall limit to one hour. Run the following from the parent’s clean detached deployment, with the full deployed SHA and a fresh attempt name.

```sh
bash outputs/symmeta-perceiver-retry-20261123-r1/train.sh "$PIN" "$ATTEMPT"
```

The launcher requires a Baige job hostname, verifies the full revision, reserves the attempt directory and runs the repository training probe with one expected GPU and exclusive-container admission. Probe and training share a one-hour deadline. The attempt is under `/mowenjie/fluffy-bassoon-2/outputs/symmeta-perceiver-retry-20261123-r1/$ATTEMPT`. The launcher records its pin, host, timings and exit code. Monitor `gpu.json` and `run.log` for device qualification and completed updates.

Each `checkpoints/perceiver_cont_warm-passive_attempt1-seed{41,42,43}.msgpack` is saved with its Stage 1 reader configuration, seed and step count and reloaded by `load_v2_checkpoint`. All three decoded factors on all 96 locked cases must agree exactly before and after restore. The reloaded parameters supply the live-worker predictions. Per-seed `training-sanity.json` entries include final loss, fit time, checkpoint digest, reload coverage, PR7’s original batched scoring path, scores from the exported factors, delta from the verified PR7 records and residuals to the rounded same-seed campaign predictions. PR7 retained aggregate tuned scores. Its per-case tuned predictions and checkpoint bytes were unavailable for historical comparison.

The final atomic `predictions.json` contains schema 1 and exactly 1,152 rows keyed by `(panel, index, seed)`. Each contains `suite`, `task_id`, `embodiment`, `tape_sha256`, and `tuned` and `symmeta` dictionaries of unrounded `yaw_deg`, `xy_scale` and integer `gripper_sign`. The receipt also reports whether all six canonical OOD yaw estimates match PR7’s published two-decimal precision. Inspect the signed per-seed differences in `training-sanity.json` before describing the refit as a numerical reproduction or launching the native comparison. Real-valued differences remain visible when input and checkpoint identity checks pass. The parent supplies the chosen attempt’s immutable path and SHA-256 to the live worker. [Exporter](export.py) · [Single-GPU launcher](train.sh)
