# Recent motion feedback lost four successful Kinova3 trials

The frozen π0.5 solved **5/6** fresh simulated Kinova3 starts with unchanged actions. The source-trained feedback controller solved **1/6**, with zero rescues and four newly failed starts. A fixed Jacobian fitted on a separate 15-control calibration solved **4/6**. Giving the feedback controller three-chunk-old motion errors solved **0/6**. Recent errors helped relative to stale errors on one cream-cheese start that the untreated policy already solved. These observations argue against continuous compensation toward this learned motion forecast.

| Task and released start | Untreated | Fixed Jacobian | Recent feedback | Stale feedback |
| --- | ---: | ---: | ---: | ---: |
| Drawer and bowl 16 | 206 | 204 | fail | fail |
| Drawer and bowl 17 | fail | fail | fail | fail |
| Two moka pots 16 | 393 | 453 | fail | fail |
| Two moka pots 17 | 410 | fail | fail | fail |
| Cream cheese and bowl 16 | 109 | 103 | 112 | fail |
| Cream cheese and bowl 17 | 104 | 97 | fail | fail |
| **Native successes** | **5/6** | **4/6** | **1/6** | **0/6** |

Numbers are controls to the first native success. Failures exhausted the released horizon. For recent versus untreated, the effect was −66.7 percentage points with four losses and zero gains. The exact two-sided discordant-pair sign probability is 0.125 under independent sign symmetry. Trials share three selected tasks, with two starts apiece and one noise draw per start. The experiment measures substantial observed harm on this panel and leaves transfer to other tasks or arms unresolved. [Per-start results, pair checks and tape hashes](native-evidence.json)

**What the controller changed.** All four conditions had the same physical reset, policy inputs, noise bank and complete first frozen action proposal for every start. Both feedback conditions shared the untouched policy's first 15 actual controls and observed response trajectory. Both first changed control 16 on every start. The recorded interventions stayed inside the normalized action bounds, respected the per-control translation norm limit and preserved the wrist and gripper commands. [Independent readback](verify.py)

The first intervention allows a direct check of the tracking objective. On controls 16–20 the six pairs still shared the previous physical trajectory and motor proposal, reconstructed with at most 1.5×10⁻⁸ absolute rounding error. Mean distance to the controller's source-fitted motion forecast was **13.73 mm untreated, 7.38 mm with recent feedback and 14.23 mm with stale feedback**. Recent feedback improved that target on five starts. Relative to the original source law before fitting the command-dependent bias, the corresponding errors were **3.81, 8.60 and 5.32 mm**. The two candidate motion references disagree about what counts as better tracking. This initial-chunk diagnostic does not identify which later contact or task phase caused failure. The native rewards do establish that improving the controller's local tracking objective did not improve completion. [First-intervention measurements](first-chunk.json) · [Diagnostic code](first_chunk.py)

Baige Distributed Training job `job-nms6mr8shrgx` exited 0 at 20:30:29 CST on Nov 23, 2026, at revision `522ccf30d18f96f4ad49104c9c862096a9db1652`. Runtime was 730 seconds on one RTX 4090 (0.203 GPU-hours). The 24 full native trajectories remain at `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-tracking-feedback-20261123-r1/gpu01`. [Registered comparison](PLAN.md) · [Source-only fit and replay](HANDOFF.md)
