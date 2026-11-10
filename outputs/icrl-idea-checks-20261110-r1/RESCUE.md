# Held lateral corrections on six natural OpenDrawer failures

A fixed −2 cm lateral goal offset rescued two of six native failures. Both scenes had negative nominal lateral misses. The prescribed rule of correcting opposite the miss recovered zero scenes, even allowing either magnitude in that direction. Choosing the best of all four offsets separately per scene still gave two successes. The development gate required at least three rescued scenes and a consistent sign rule.

| Scene | Nominal lateral miss (cm) | Native two-finger grasp | Offsets that produced a native opening | Opening control |
| --- | ---: | ---: | --- | ---: |
| 982100 | −5.49 | 282 | None | — |
| 982102 | +3.67 | — | None | — |
| 982104 | −0.94 | — | −2 cm | 514 |
| 982105 | +0.68 | — | None | — |
| 982106 | +2.51 | — | None | — |
| 982107 | −2.74 | — | −2 cm | 510 |

Miss is the median of the four archived signed approach measurements. Its sign was consistent across all four measurements on each scene. Each corrected branch replayed the current nominal prefix up to the first approach within 10 cm. The offset lasted until the first grasp or 100 controls, whichever came first. All six zero continuations failed. Physical state at every step through the decision and the public decision observation matched the nominal on all 30 branches. [Per-episode records](rescue03/events.jsonl) · [Measurements by scene](DRAWER-SUMMARY.json).

**Replay checks.** Replaying the archived commands reproduced all measured simulator states and terminal hashes on all eight screened scenes. Fresh policy inference reproduced all eight native outcomes and six terminal hashes. On the other two, the first command differences appeared at zero-based controls 255 and 245, with largest component differences of 0.000183 and 0.001953 on those commands. Four of the six zero continuations also reproduced their fresh nominal terminal hashes. The initial attempt had stopped on a late trajectory hash mismatch before any offset branch. [Initial attempt](rescue02/events.jsonl).

**Video review.** Scene 982100 acquired the handle and moved the drawer 30.1 cm without meeting native success. The other five nominal failures never registered a two-finger grasp. The two successful corrections reached grasps at controls 454 and 455, well after the offset window ended, and moved the drawer 31.9 and 31.5 cm. The intervention changed the subsequent closed-loop trajectory. These six scenes provide no evidence that a previous episode is needed to choose the correction. The comparison frames are sampled every five controls. [Six nominal approaches](NOMINAL-FRAMES.jpg) · [Paired frames near grasp and opening](RESCUED-FRAMES.jpg) · [982104 zero](rescue03/982104-zero_repeat.mp4) · [982104 corrected](rescue03/982104-offset-2cm.mp4) · [982107 zero](rescue03/982107-zero_repeat.mp4) · [982107 corrected](rescue03/982107-offset-2cm.mp4).

Baige Pro6K job `job-x15mur4ftoja` used source `0b1c3bde`. The worker completed 46 episodes and 26,090 controls in 2,470 seconds on one GPU (0.686 GPU-hours). [GPU probe](rescue03/gpu.json) · [Worker clock](rescue03/exit.json). The scheduler recorded success and 41m43s (0.695 GPU-hours). [Scheduler receipt](rescue03-close.png).
