# A single reset image identified physical camera roll on seven of eight tasks

**Physical Layer B Gate 0 qualified.** A frozen 16 × 16 RGB template reader identified the MuJoCo agentview camera's roll on **21/24 development images (87.5%)**. It got all three orientations right on seven tasks and all three wrong on the eighth. Replacing every input with that task's nominal image gave **8/24 (33.3%)**. Static appearance therefore exposes roll before any public motion on this panel. The prespecified perfect score across all eight tasks was missed. We stopped at Gate 0. [Registered plan](PLAN.md) · [Predictions and distances](cpu01/result.json)

| Actual camera roll | Predicted 0° | Predicted +90° | Predicted −90° |
| --- | ---: | ---: | ---: |
| 0° | 7 | 1 | 0 |
| +90° | 0 | 7 | 1 |
| −90° | 0 | 1 | 7 |

All errors occurred on the living-room basket task, `libero_10` task 0. The reader predicted +90°, −90° and +90° respectively for its nominal, +90° and −90° images. Across all 24 exposures the mean absolute circular error was 18.75°. The smallest gap between the two lowest template mean squared errors was 184.88 in squared 8-bit intensity units, on that task's nominal image. [Nominal frame](cpu01/dev-libero_10-0/roll+0.png) · [+90° frame](cpu01/dev-libero_10-0/roll+90.png) · [−90° frame](cpu01/dev-libero_10-0/roll-90.png)

The native adapter settled one BASE reset per task. Each exposure then changed only the external camera's model quaternion and refreshed the rendered observations at public control zero. Positive angles use the camera's forward optical axis (local −Z). Across 64 readbacks, including the return to nominal on every reset, `cam_xmat` relative to `robot0_base` recovered the requested roll within 1.43 × 10⁻¹⁴ degrees. The camera center displacement was zero and the maximum optical-axis vector change was 2.50 × 10⁻¹⁶ or less. The complete MuJoCo integration state, all body poses, five public proprio fields and wrist RGB matched their released reset bytes at every exposure. Restoring nominal reproduced all 16 original agentview images exactly. Rolled development images differed from the same task's nominal at 99.61–99.96% of pixel locations. [Independent readback](cpu01/verification.json) · [Example of raw camera matrices and state hashes](cpu01/dev-libero_10-0/readback.json)

The reader averaged eight fitting images per angle and used one held-out agentview image per prediction. The templates were frozen before the first development reset. A second calculation on Baige recovered the templates and predictions from all 48 saved PNGs. The evidence covers one official BASE initial state per task (fitting 0, development 13), three discrete camera rolls and zero post-reset actions. Action/local-frame yaw and policy success are separate questions. No official test outcomes were accessed. [Task IDs, seeds in the per-task records, and runtime versions](cpu01/protocol.json) · [Frozen templates](cpu01/templates.npy)

Baige Linux CPU attempt `cpu01` ran detached registration `4d9a2eaeee4038fc6d9a94219fd79d841bd6105b` on LIBERO-Pro2.0 `e5500c690c1498eca1b7050350654964190cdf70`. Mesa llvmpipe rendered the images. The job exited 0 after 136 seconds, 20:03:00–20:05:16 Shanghai on November 19. Usage was **0 GPU-hours**. [Exact experiment source](gate.py) · [Launch and source hashes](cpu01/launch.txt) · [Exit record](cpu01/exit.txt)
