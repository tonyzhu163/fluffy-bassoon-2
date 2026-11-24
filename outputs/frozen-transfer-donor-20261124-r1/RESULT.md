# A different-layout IIWA tape rescued one Kinova3 failure

**The successful same-layout IIWA commands rescued neither failed Kinova3 start.** Kinova3 timed out on its own recordings for layouts 36 and 37, and the CPU own-tape controls reproduced both timeouts. The matched IIWA tapes ran out after 248 and 251 controls without reward. But on Kinova3 layout 37, the IIWA tape from **layout 35** earned the native reward at control **182 of 184**. It was the only successful transfer among the eight combinations of four IIWA donors and two failed Kinova3 starts. All four donors exhausted their commands on layout 36. [Registered question and comparisons](PLAN.md) · [All 32 native endpoints](evidence/cases.csv)

## Paired outcomes

The task was to open the top drawer and put the bowl inside. Each entry gives the native outcome and last executed control, numbered from one. A win is the first native success and reward 1. A timeout is native truncation at the 300-control horizon and reward 0. A tape end means the final available action returned reward 0 with both terminal flags still false. Each cyclic donor came from the opposite body at the next row, wrapping 37 to 34.

| Target | Layout | Original live raw | CPU own raw | Opposite body, same layout | Opposite body, cyclic layout |
|---|---:|---|---|---|---|
| Kinova3 | 34 | Win 225 | Win 225 | Win 170 | Win 179 |
| Kinova3 | 35 | Win 186 | Win 186 | Win 177 | Tape end 248 |
| Kinova3 | 36 | Timeout 300 | Timeout 300 | Tape end 248 | Tape end 251 |
| Kinova3 | 37 | Timeout 300 | Timeout 300 | Tape end 251 | Tape end 171 |
| IIWA | 34 | Win 171 | Win 171 | Win 225 | Win 186 |
| IIWA | 35 | Win 184 | Win 184 | Win 186 | Timeout 300 |
| IIWA | 36 | Win 248 | Win 248 | Timeout 300 | Timeout 300 |
| IIWA | 37 | Win 251 | Win 250 | Timeout 300 | Win 222 |
| **Successes** | **8 starts** | **6/8** | **6/8** | **4/8** | **3/8** |

On Kinova3 the own, matched and cyclic conditions succeeded on 2, 2 and 1 of four starts. The only matched advantage over cyclic occurred at 35, where the own arm already succeeded. On IIWA the counts were 4, 2 and 2. Each successful Kinova3 donor transferred to its own layout on IIWA. The two same-layout Kinova3 donors that timed out at recording also timed out on IIWA. The reverse-direction comparisons therefore vary donor success and length as well as the layout.

Across both bodies, matched versus own produced zero rescues, two losses and six ties (−25 percentage points). Matched versus cyclic produced two gains, one loss and five ties (+12.5 points). The four paired layout differences for matched versus cyclic were 0, +100, 0 and −50 points. The descriptive 95% intervals from all 256 ordered resamples of whole layouts were −50 to 0 points for matched minus own and −37.5 to +75 points for matched minus cyclic. Four selected layouts from one task support a case-level result. [Per-arm contrasts and layout effects](evidence/summary.json)

## Four successful IIWA tapes on every Kinova3 start

| Kinova3 start | IIWA 34 (171 controls) | IIWA 35 (184) | IIWA 36 (248) | IIWA 37 (251) |
|---:|---|---|---|---|
| 34 | Win 170 | Win 179 | Tape end 248 | Tape end 251 |
| 35 | Tape end 171 | Win 177 | Tape end 248 | Tape end 251 |
| 36 | Tape end 171 | Tape end 184 | Tape end 248 | Tape end 251 |
| 37 | Tape end 171 | **Win 182** | Tape end 248 | Tape end 251 |

Perfect hindsight selection among these four complete tapes could solve **3/4** tested Kinova3 starts, gaining **1/2** original failures over the own replay. The layout-35 donor alone attained that ceiling on this panel. Its 184 commands supplied less exposure than the unsuccessful 251-command matched donor on layout 37. This establishes a concrete transferable command sequence. The matrix supplies no prospective way to identify it. Exact scene matching missed the rescue, and exhausting all four tapes at 36 leaves zero further selection headroom on that local start for this finite library. [Per-case submitted-action digests, native flags and reward sequences](evidence/cases/)

Every successful source episode stopped before the horizon. The four IIWA tapes supplied 171–251 controls. Kinova3 successes supplied 225 and 186 controls, while failures supplied 300. A tape that ended without reward leaves the effect of further actions unknown. Across the 32 branches there were 14 native successes, six native horizon truncations and 12 tape ends. A hypothetical extended sequence or closed-loop replan falls outside the measured comparison.

## Reset fidelity, commands and provenance

- Each target used Goal 3 BASE, seed 42710300 + layout and the published 50 × 79 state bank. Each row had a distinct released state and scene fingerprint. The GPU arms shared the official row, pre-settle scene, BDDL, perturbation, random seed and full original policy noise bank within each layout. Each bank and each original query noise digest was regenerated. On the CPU all branches for a given body/layout agreed exactly on the full reset record and all initial public observations. All 32 matched on the first reset attempt. All eight CPU physics fingerprints also matched the earlier CPU03 qualifier. [Runtime record](evidence/cases/runtime-30.json) · [Original-file manifest](inputs/manifest.json)
- Own replay reproduced the native binary outcome on **8/8** starts and the complete reward sequence on **7/8**. IIWA layout 37 reached success one control earlier. On all eight starts the CPU and GPU settled physics hashes differed, as did their post-settle scene hashes. The per-query proprioception digests also show trajectory differences. For example, CPU and GPU public end-effector position bytes matched on one of 60 own-tape queries for Kinova3 36 and 50 of 60 for 37. These counts measure exact equality and cannot quantify physical displacement. Thus the finite-library ceiling applies to these exact CPU starts. Small numerical differences around contact could change a counterfactual outcome on the original host. [Own-tape trajectory and timing checks](evidence/summary.json)
- The cue recorder stored the executed prefix of the bounded commands sent to the simulator. The original raw motor used identity yaw, unit scale and positive gripper sign; the nominal worker passed each submitted command to the physical arm unchanged. The pinned native ArmEnv converted to float32 and passed a copy to the underlying LIBERO step. Replay read only the stored submitted actions. At every underlying step it verified the seven float32 components byte-for-byte against the donor prefix and verified the native step counter. Rendering after reset returned a blank RGB buffer because the tape was fixed. Physical integration, OSC, gripper and native reward ran normally. [Recording interface](../frozen-transfer-cue-20261124-r1/cue.py) · [Original worker](../libero-xemb-ur5e-sysid-20261120-r1/ur5e_sim_worker.py) · [Physical interface](../libero-crossarm-20261120-r1/arm_env.py) · [Replay](replay.py)
- Baige development CPU cores 30 and 31 ran 7,284 controls in **228 seconds**, 11:45:17–11:49:05 Shanghai, 24 November 2026. Mesa llvmpipe rendered the initial observations. GPU use was zero. The replay pin was `c77f99eac2424c7ec1d73633595b71a11c833bfa`, physical interface `a8a997e7d06bb131d3f9710204c58e106f0f0904`, and LIBERO-Pro2.0 release `e5500c690c1498eca1b7050350654964190cdf70`. The frozen π0.5 commands came from job `job-wnafkdyqfdu5` at `e15eefb3f76e475acdf0795e2e17ba0e1ee46fae`. Both workers verified the eight inputs against the original job files; the portable readback rehashed the consumed prefix of every case and checked the per-step flags and rewards. [Eight original raw recordings](inputs/raw-tapes.tar.gz) · [Launch script](run.sh) · [Exit receipt](evidence/exit.txt) · [Portable readback](readback.py)
