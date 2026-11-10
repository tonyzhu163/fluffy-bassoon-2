# Which native failures repeat?

On CloseDrawer, every one of eight development scenes returned the same native outcome on four different motor-noise seeds. Five scenes always closed and three always failed. OpenDrawer also had strong scene dependence. The six successes apiece on CloseFridge and TurnOnMicrowave were more dispersed.

| Native task | Successes / 32 | Successes per scene, seeds 983000–983007 | Scenes with 0/4 | Expected 0/4 under independent attempts | Scenes with 4/4 | Scene ICC |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CloseDrawer | 20 | 0, 4, 4, 0, 0, 4, 4, 4 | 3 | 0.16 | 5 | 1.000 |
| OpenDrawer | 13 | 0, 4, 4, 0, 0, 1, 0, 4 | 4 | 0.99 | 3 | 0.885 |
| CloseFridge | 6 | 3, 0, 0, 1, 1, 1, 0, 0 | 4 | 3.49 | 0 | 0.222 |
| TurnOnMicrowave | 6 | 1, 0, 2, 1, 0, 1, 1, 0 | 3 | 3.49 | 0 | −0.067 |

For the independent-attempt reference, the expected number of always-failing scenes is eight times the fourth power of the observed task failure rate. The one-way intraclass correlation (ICC) estimates how strongly repeats on a scene agree, relative to the variation across scenes. With eight scenes the estimate is imprecise and can fall below zero. CloseFridge and TurnOnMicrowave each had an 81% failure rate, enough to give about 3.5 always-failing scenes by chance. The earlier AD readout cleared history under a different policy and sampled different scenes. Its ICCs ranged from 0.05 to 0.16. [Machine-readable results](PERSISTENCE.json) · [Readout](persistence_readout.py) · [AD reference](STAGE1-REPEATABILITY.tsv).

## What this means for the experiment

CloseDrawer is the first task to inspect for a correction. Its three repeatedly failing starting scenes are 983000, 983003 and 983004. The five consistently successful scenes supply controls for whether a fixed correction would hurt. The matched OpenDrawer tests show why that comparison matters. A fixed −2 cm offset achieved every observed rescue (2/6), and normal and doubled post-grasp translation succeeded on the same 6/12 conditions. In both tests, letting an oracle choose separately for each condition added zero successes. [Natural failures](RESCUE.md) · [Damping and effort](STIFF.md).

Hold the 100-trial collection. First establish on development cases that distinct conditions call for distinct successful actions and that selecting per condition improves on the best fixed alternative. Before interpreting that as adaptation, test whether the previous attempt supplies information beyond the current observation on fresh query scenes sharing the hidden property. Repeats on an identical scene can expose a stable obstacle even when its cause is already visible.

## Protocol and verification

Each of the four frozen native Human300 π0.5 jobs ran 32 episodes and exited successfully. All 128 registered task–scene–repeat combinations are present, covering 62,756 delivered controls. Noise seed was scene + 700000 + 1000 × repeat. Every repeat on a given task and scene shared one exact simulator and public-observation start. After an early CloseFridge attempt exposed a reset mismatch, the qualified jobs selected the first starting configuration observed twice for each registered scene before taking any action. All scene seeds were retained. The logs record 190 candidate resets, 128 selected starts and a maximum of 11 candidates for a repeat. Image variation accounted for the observed within-search differences. [Protocol](PLAN.md) · [Reset and episode evidence for closing](closedrawer02/events.jsonl), [opening](opendrawer02/events.jsonl), [fridge](fridge02/events.jsonl) and [microwave](microwave02/events.jsonl).

| Work on Baige Pro6K, one GPU per job | Job | Pinned source | Worker seconds | Scheduler allocation |
| --- | --- | --- | ---: | ---: |
| CloseDrawer persistence | job-7s99ojrfztfz | ad9003d7 | 1,381 | 23m12s |
| OpenDrawer persistence | job-gdoxow8fa5ii | ad9003d7 | 1,566 | 26m10s |
| CloseFridge persistence | job-8ag444u5qo0v | ad9003d7 | 1,859 | 31m11s |
| TurnOnMicrowave persistence | job-98bqpnji9q24 | ad9003d7 | 2,033 | 34m12s |
| Natural-failure rescue | job-x15mur4ftoja | 0b1c3bde | 2,470 | 41m43s |
| Damping and effort | job-19vkb4r2o74v | 9c0036ea | 941 | 18m04s |
| Early rescue stopped on terminal divergence | job-m1eub51tsxw8 | 9c0036ea | 97 | 2m11s |
| Early persistence stopped on reset mismatch | job-b77hmbv1y3za | 9c0036ea | 102 | 2m10s |

The persistence panel used 1.900 GPU-hours of worker time. The three checks together used 2.903 worker GPU-hours and 2.981 scheduler GPU-hours, including both early exits. Queued requests were resized for capacity. A 4090 request was withdrawn while queued. All scored episodes ran on Pro6K. [Scheduler receipt](PERSISTENCE-close.jpg) · [First persistence attempt](persistdoors01/events.jsonl) · [First rescue attempt](rescue02/events.jsonl).
