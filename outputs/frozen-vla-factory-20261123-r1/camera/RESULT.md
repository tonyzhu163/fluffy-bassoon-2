# An independent camera angle reduces frozen-reader yaw error

One fixed optical-roll estimate per camera reduced command-yaw MAE from **92.45° to 27.64°** on synthetic quarter-turn views of 56 held-out UR5e supports. The estimated correction used one separate calibration trajectory. Giving the same readout the exact roll yielded 27.57°. Known camera azimuth reduced error from **28.98° to 7.61°** on eight physically rerendered orbit trajectories. All results measure offline command-interface identification.

| Matched panel | Supports / tasks | Raw yaw MAE | Known angle | Estimated angle | Opposite sign | Estimated reduction, 95% task interval |
|---|---:|---:|---:|---:|---:|---|
| Full frame, 0° | 56 / 7 | 6.60° | 6.60° | 6.71° | 6.60° | −0.11° [−0.17, −0.05] |
| Full frame, ±90° | 56 / 7 | 92.45° | 27.57° | 27.64° | 152.43° | 64.81° [59.52, 69.30] |
| 1.4× zoom, 0° | 56 / 7 | 27.68° | 27.68° | 27.75° | 27.68° | −0.07° [−0.32, 0.16] |
| 1.4× zoom, ±15° | 56 / 7 | 25.59° | 19.23° | 19.40° | 37.25° | 6.19° [4.37, 7.81] |
| 1.4× zoom, ±30° | 56 / 7 | 37.43° | 19.32° | 19.60° | 66.71° | 17.83° [14.22, 21.65] |
| Physical orbit, 0° | 8 / 2 | 7.76° | 7.76° | — | 7.76° | — |
| Physical orbit, ±30° | 8 / 2 | 28.98° | 7.61° | — | 58.98° | — |

**Inputs and pairing.** The motor was the archived frozen OpenPI `pi05_libero` policy. The comparison reuses the native seed-41 SymMeta yaw readout on each support. This reader uses image motion, public dz and issued commands. All arms receive the same trajectory of three five-control chunks. Only the additive camera term changes. Digital roll uses the known displayed rotation, while physical orbit uses the relative world-vertical rotation of the measured camera-to-world matrices. The opposite-sign control negates that term. Physical orbit used fresh simulator trajectories that repeated stored command means and rendered three views at each boundary. No camera change was delivered to the motor. [Motor wrapper](../../libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py) · [Archive and frozen reader provenance](../../symmeta-camera-command-20261123-r1/SCORING.md) · [Physical replay](../../symmeta-camera-command-20261123-r1/ORBIT.md)

**Estimated roll.** Each tested mount provides measured brightness flow from one 15-control calibration trajectory and an independent world TCP measurement from UR5e joint FK. A camera projection fitted previously on nominal Panda/Sawyer motions predicts upright pixel displacements. The rotation aligning the observed to predicted image motion is added to the reader output. The estimator receives no numerical camera or command-yaw label. The first task's slot 0 supplies calibration throughout. Excluding that entire task leaves 56 OOD supports with distinct task identities from calibration. The raw and known-angle arms use zero calibration controls. The estimated arm spends 15 once per fixed mount, then uses the same 15 controls per test support as the controls. The angle is shared across all seven test tasks. It requires the camera setup to stay fixed from calibration through scoring. [Source projection](../../symmeta-camera-command-20261123-r1/RUNTIME.md) · [FK qualification](../../symmeta-camera-command-20261123-r1/WORLD.md) · [Plan and falsifier](PLAN.md)

**Uncertainty and failures.** The intervals resample whole tasks after averaging the two signs on each trajectory. They are conditional on the named calibration donor and these transform panels. On the full 64-support, eight-task panel, known quarter-turn correction reduced MAE by 66.05° [60.79, 70.56]. The two physical-orbit task reductions were 21.94° and 20.80°. Two tasks provide little evidence about variation across new tasks. Across all eight choices of a single calibration donor, each evaluated on the other seven tasks, the quarter-turn reduction ranged from 64.04° to 68.84°. For moderate rolls, seven donors helped on average and Goal 6 *increased* MAE by 2.52°. On that donor a −30° displayed roll was estimated as +12.59°. This is an estimate of one roll component from a reference motion. The experiment does not estimate physical orbit, camera translation, or orientation on an unreferenced unknown view.

**Architecture decision.** Use a separately calibrated orientation input when a mount is stable. With a known physical azimuth the post-readout correction recovered the nominal-view level on the two orbit tasks. For large optical rolls, prefer testing image canonicalization before the frozen motion reader. Even the exact angle added after readout left 27.57° MAE versus 6.60° upright, and the archive verified that exact quarter-turn inversion before the reader recovers the original pixels and estimates. If trusted world FK is available on every target trajectory, the archived FK-only fit already gives 1.97° MAE on these 56 supports. A referenced visual reader addresses the narrower case where that measurement is available for calibration alone. Native task success, π₀ transfer and camera changes seen by the motor remain unmeasured here. [Original inverse and FK controls](../../symmeta-camera-command-20261123-r1/RESULT.md)

## Reproduce

Base revision `757a44cd4374833cad790852df8b87ddf412e638`. The committed compressed JSON contains the support NPZ hashes, per-chunk measurements, frozen predictions and physical camera matrices. Raw UR5e support NPZs reside on Baige. This scorer reads the compact archive locally, checks the support joins and uses command-yaw labels only to score predictions. The collector revision was `5dd5fa914bbbfe25b9cf4ca5fa504c8a472e997b`, the frozen reader digest starts `c12bace5cb39`, and the ordered support-set digest starts `8729fa68343e`. [Source and full digests](../../symmeta-camera-command-20261123-r1/SCORING.md) · [Scorer](screen.py) · [Metrics by angle, task and donor](metrics.json)

Run from the repository root with this lane commit applied.

```sh
lane=outputs/frozen-vla-factory-20261123-r1/camera
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 uv run --no-sync python "$lane/screen.py" --output /tmp/camera-metrics.json
cmp /tmp/camera-metrics.json "$lane/metrics.json"
```

Local CPU receipt `camera-reference-r1`. Python 3.12.16, NumPy 2.5.4, exit 0. Replay took 0.42 s wall time and used 50,908 KiB peak RSS. Byte-identical metrics SHA-256 `2653c6fa85fefc8894f2c338df4f5ee00da631f0662bc470c3e6c0f6e881b0fe`. Zero GPU-hours, zero motor updates, zero new native episodes.
