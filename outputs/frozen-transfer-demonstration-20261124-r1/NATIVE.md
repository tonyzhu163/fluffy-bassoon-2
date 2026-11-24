# A source approach rescued four tuned drawer failures

The source-selected IIWA 34 approach followed by frozen π0.5 with the standard Kinova3 Perceiver correction succeeded on **6/8 reused layouts**, compared with **2/8 tuned** and **3/8 raw**. It preserved both tuned successes and rescued four of the six tuned failures. Three of those four layouts also failed under raw control. The primary paired difference was **+50 percentage points**. The secondary donor also finished 6/8, with a different set of successes. [Registered plan](PLAN.md) · [Native step readback](evidence/native01/portable-readback.json)

## Matched native outcomes

A number gives the one-based control on which the native task returned success. A dash means the environment timed out at 300 controls with zero return. Each row is one official Kinova3 BASE start on `libero_goal/3`, with seed 42710300 plus the layout number.

| Reused layout | Raw π0.5 | Tuned | IIWA 34 → tuned | IIWA 35 → tuned |
|---:|---:|---:|---:|---:|
| 40 | — | — | 220 | 236 |
| 41 | 171 | 175 | 175 | 187 |
| 42 | 181 | 169 | 179 | — |
| 43 | — | — | 169 | 197 |
| 44 | — | — | — | — |
| 45 | — | — | 218 | 194 |
| 46 | 200 | — | 176 | 196 |
| 47 | — | — | — | 196 |
| **Native successes** | **3/8** | **2/8** | **6/8** | **6/8** |

| Paired comparison | Rescued layouts | Lost layouts | Ties | Difference | 95% paired layout bootstrap |
|---|---|---|---:|---:|---:|
| **IIWA 34 vs tuned (primary)** | **40, 43, 45, 46** | **None** | **4** | **+50 pp** | **+12.5 to +87.5 pp** |
| IIWA 35 vs tuned | 40, 43, 45, 46, 47 | 42 | 2 | +50 pp | 0 to +87.5 pp |
| IIWA 34 vs raw | 40, 43, 45 | None | 5 | +37.5 pp | +12.5 to +75 pp |
| IIWA 35 vs raw | 40, 43, 45, 47 | 42 | 3 | +37.5 pp | −12.5 to +87.5 pp |
| Tuned vs raw | None | 46 | 7 | −12.5 pp | −37.5 to 0 pp |

Intervals resample the eight whole matched layouts with replacement. They describe sensitivity to the panel composition. On the primary comparison only four layouts were discordant. All four favored the hybrid, giving a two-sided exact sign-test probability of 0.125 under sign symmetry. These eight fixed starts on one task and one target embodiment support a rescue on this panel. They provide limited information about a new task or release. [Paired estimates](evidence/native01/result.json)

## What the approach and handoff established

The primary rule chose the shortest successful archived raw IIWA demonstration using source information alone. IIWA 35 had been selected earlier on four Kinova development layouts. Both donors were fixed for every target row. The hybrid used the original seven-channel issued actions for 105 or 115 controls, through the last complete five-control block before the first positive gripper command. The two first-positive indices were 106 and 119, zero-based. The grip switch defines a reproducible cut, while the precise physical phase at that instant can differ across layouts. The tuned correction came from the 15 controls of separate layout 11, PR7 reader seed 41. [Donor and support provenance](evidence/native01/origin.json)

All 16 hybrids reached handoff. The first resumed queries were global replans 21 and 23. Each saw the target cameras and current proprioception from the immediately preceding native step, used the original released instruction and corresponding noise, and produced all five commands through the ordinary tuned transform. Subsequent queries used updated target observations. Across the panel, 1,760 source controls matched the archive byte for byte. Successful hybrids needed a further 64–115 native controls with donor 34 and 72–121 with donor 35. Within each quartet, the full reset, opening motor inputs and opening frozen proposal matched on the first reset attempt. All four branches used the same loaded checkpoint and time-indexed noise. [Audited episodes](evidence/native01/episodes.tar.gz) · [Full numerical readback](evidence/native01/host-readback.json)

The primary donor's gains show that a fixed task-specific approach can bring the frozen target controller to states from which the native drawer task completes. The secondary donor exchanged a success on 42 for one on 47 relative to the primary donor. Layout 44 failed in all four branches. The experiment isolates the effect of substituting the prefix for ordinary tuned actions, including the target states those actions produce. It cannot separate which prefix motions mattered, or estimate the benefit of feedback over simply continuing the donor on an identical settled start.

The separate CPU whole-tape validation completed 3/8 with tape 34 and 2/8 with tape 35. It used these same released layout IDs and seeds. All eight nominal release fingerprints and pre-settle scene fingerprints matched this job, while all eight initial physics fingerprints differed across hosts. Unsuccessful CPU tapes ended when their recorded actions ran out. Those data do not supply a same-start or same-horizon contrast with the distributed hybrids. [CPU whole-tape results](../frozen-transfer-donor-validation-20261124-r1/RESULT.md) · [Compared start fields](evidence/native01/cross-host-starts.json)

## Execution and portable evidence

Baige Distributed Training job `job-mbgx2xjo8rqm`, attempt `gpu01`, used the detached pin `ef165714d626305fa756e10fac2b5f7c45446f1c`. The in-job probe passed on one RTX 4090. The launcher ran from 12:17:39 to 12:30:06 Shanghai on November 24, 2026, exited zero after **747 seconds** (0.208 launcher GPU-hours), and recorded **32 native terminals, 7,739 physical controls and 1,555 policy calls**. All four modes queried the motor even while a donor was executing. [Launch and qualification](HANDOFF.md) · [GPU probe](evidence/native01/gpu.json) · [Launcher receipt](evidence/native01/exit.txt) · [Native log](evidence/native01/run.log)

The archive retains all 32 original episode JSONs. The portable scorer independently reads the actual native step rewards, terminal flags and action arrays, checks both source prefixes and the opening matches, and reconstructs the counts and intervals. All 32 member hashes and all five contrasts matched the live writer. On the Baige development host the default BLAS kernel fused XY arithmetic, producing 1,437 float32 component differences with maximum absolute error 1.19 × 10⁻⁷ when recomputing tuned actions. The job's actions matched separate float32 multiply and add throughout. Repeating the unmodified full verifier with a non-fused OpenBLAS kernel reproduced every physical action byte, all noise digests and all native outcomes. This numerical difference arose during readback. [Member hashes](evidence/native01/original-episodes.sha256) · [Host verification command](HANDOFF.md) · [Independent audit](READBACK.md)

To reproduce the core native scores from the repository root with standard Python, run

```bash
python3 outputs/frozen-transfer-demonstration-20261124-r1/portable.py outputs/frozen-transfer-demonstration-20261124-r1/evidence/native01/episodes.tar.gz
```
