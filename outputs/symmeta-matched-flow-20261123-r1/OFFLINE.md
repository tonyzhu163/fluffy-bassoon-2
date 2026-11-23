# Matched-source passive yaw estimation on UR5e × π₀.₅

The frozen seed-41 visual reader reduced mean absolute yaw error by **2.63°** against a six-coefficient linear flow reader on the 64 held-out task–transform pairs. The 95% paired task-bootstrap interval was **1.64° to 3.65°**, and all eight task means favored the learned reader. The compositional transforms accounted for most of the gap. For continuous angles the observed difference was 0.30° and the interval spanned zero. [Registered comparison](PLAN.md) · [Metrics with intervals and per-task means](evidence/metrics.json)

| Transform panel | Pairs | Learned MAE | Linear MAE | Linear minus learned, 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Held-out transforms | 64 | 6.53° | 9.16° | +2.63° [1.64, 3.65] |
| All transforms | 96 | 6.37° | 8.93° | +2.56° [1.84, 3.24] |
| Compositional | 32 | 6.20° | 11.16° | +4.96° [2.52, 7.80] |
| Continuous angle | 32 | 6.87° | 7.17° | +0.30° [−2.40, 2.90] |
| Familiar | 32 | 6.06° | 8.47° | +2.41° [0.37, 4.42] |

Each observation is the wrapped absolute yaw error on one official test task and one transform. Each interval uses 100,000 draws of eight tasks with replacement, preserving both methods and every transform within each sampled task. The bootstrap seed was 20261123. The eight tasks, fixed camera, single learned checkpoint and single linear fit limit inference. Native second-attempt success is a separate live endpoint.

| Suite / task | Held-out learned MAE | Held-out linear MAE | Held-out difference | All-transform difference |
| --- | ---: | ---: | ---: | ---: |
| libero_10 / 5 | 6.09° | 10.06° | +3.97° | +2.70° |
| libero_10 / 8 | 5.39° | 9.33° | +3.94° | +3.39° |
| libero_goal / 3 | 7.61° | 12.59° | +4.98° | +2.65° |
| libero_goal / 6 | 6.03° | 7.17° | +1.14° | +2.65° |
| libero_object / 1 | 5.90° | 7.34° | +1.45° | +0.61° |
| libero_object / 8 | 5.84° | 7.79° | +1.95° | +2.82° |
| libero_spatial / 0 | 7.24° | 10.21° | +2.97° | +4.15° |
| libero_spatial / 8 | 8.17° | 8.82° | +0.65° | +1.50° |

## Source fit and freeze

The archived collection ran at `d597ee0f1e99937bbeb2705132d709b514a4b5c2`. Its manifest SHA-256 was `37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076`, the manifest recorded in the historical seed-41 checkpoint trained at `d563845717e1f9312878b228328908484c4752f9`. Filtering that manifest to its training split selected 672 Panda and 672 Sawyer passive traces by archive order. Both arms covered the same 24 official training tasks, four resets and seven transforms. All 1,344 identities were unique, each trace had 15 controls in three valid chunks, and the saved membership file records the path and SHA-256 of every tape. [Source membership](evidence/source-membership.json.gz) · [Fit and source hashes](evidence/fit.json)

The linear reader maps two pooled flow components in pixels and measured vertical displacement in metres to a planar vector with a 3 × 2 matrix and zero intercept. For each frame pair the repository's `dense_flow_image_motion` computes grayscale OpenCV DIS MEDIUM. Pixels with motion magnitude at least the larger of 0.15 pixels and 25% of peak magnitude enter a squared-magnitude weighted mean of the flow vectors. The learned reader uses DIS FAST maps alongside intensity, gradient and position features and learned spatial pooling. The comparison tests the complete visual readouts under matched source traces, yaw labels and target alignment. Several visual components differ, so the observed gap cannot be assigned to the network architecture alone. [Pooling implementation](../../src/fluffy_bassoon/xemb/baselines.py) · [Frozen fit and scoring code](offline.py)

We scaled each input dimension by its source RMS. Least squares on issued horizontal commands rotated by the trace yaw supplied the initialization. Full-batch BFGS then minimized the mean trace loss `1 − cos(estimated yaw − labeled yaw)` using the normalized equal-vote alignment. The fit received source commands and one yaw label per trace. It used neither a measured world XY target nor target calibration. BFGS converged in 12 iterations, reducing source loss from 0.07499 to 0.06443. In-sample MAE was 13.49°. The saved matrix acts directly on unscaled pixels and metres. [Coefficients, scales and optimizer record](evidence/fit.json)

The fit and scoring implementation was committed as `86df380f53037eb914b22fa0e21b074f52eadf9b` and deployed by Git bundle to a fresh detached checkout on Baige. The resulting coefficients and source identities were committed as `7d2f21d7c0620dfc43ac010a1e2b01789ebf8304` before target scoring. The scoring process checked the fit digest against that Git object. Fit SHA-256 was `da955481c9471c8609688f1f1ab48e56fcb673ccbdbb558e5f3be76dac6cfb99`. Both phases ran over SSH on Baige CPU with GPU visibility disabled, OpenCL off and one OpenCV thread. Fit took 10.10 s and scoring took 0.55 s, excluding setup and verification. The checkout and raw run are under `/mowenjie/fluffy-bassoon-2/outputs/symmeta-matched-flow-20261123-r1/` in `offline-src-86df380f5303/` and `offline-cpu01/`. [Deployment and run log](evidence/offline-run.log)

## Target identities, alignment and live handoff

For each official support we joined the zero-roll row from the historical camera audit to the passive tape using index, suite, task, transform name and support SHA-256. We took `vectors.model_visual_xy` from that row. An independent CPU pass through the original seed-41 checkpoint reproduced all 288 upright chunk vectors exactly. Each reader produced its planar vectors before accessing the issued horizontal commands for alignment. For each of the three five-control chunks, we normalized both the issued XY and estimated XY to unit length and used one vote for their relative angle. The yaw estimate is the circular mean across those votes. All 96 cases supplied three nonzero votes per method and identified both yaws. A second implementation of the circular mean agreed within 2.85 × 10⁻¹⁴ degrees. [Target vectors, votes and signed predictions](evidence/target-details.json.gz) · [Checkpoint, export and panel identities](evidence/prediction-provenance.json) · [Verification log](evidence/offline-run.log)

The live handoff is a compact JSON object with one `rows` list, ordered by index 0–95. Every row contains `index`, `suite`, `task_id`, the string `embodiment`, `support_sha256`, `learned_yaw_deg` and `linear_yaw_deg`. Its SHA-256 is `c6828c9c848ccd0ce10bc89a5c91356b66c58619b08cbc07ba15d2af7608e191`. Identical bytes are available at `evidence/predictions.json` under the local and Baige run roots. All 96 indices, compound IDs and support hashes also matched the pinned seed-41 reader export (SHA-256 `b740827bcb8bf8d1f90cc86648772e8a6b9dbce5dd6e7fffeaa88a75a7deb654`). That export supplies the shared full-precision `scale_hat` and `grip_hat` for the live worker. [Yaw predictions](evidence/predictions.json) · [Seed-41 scale and gripper export](../symmeta-fk-ur5e-20261123-r1/evidence/reader.json)
