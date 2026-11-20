# Native execution on Baige

The contact replay at `692500d1` inspected ten controls in the preceding Sawyer failure. The plate positions matched the original tape at every sampled control. At control 82 the palm contacted the plate and table. From 90 through 297 the sampled contacts showed neither plate contact nor an arm–cabinet contact. The first outside-edge attempts recorded the palm bearing 37–40 N against the table in two approaches. This led to targeting the lowest collision surface of the whole gripper. [Replay](inspect_contact.py) · [Search controller](paddle.py)

The native camera-free search on Pro6K job `job-hobqjf487g5p` found one successful layout-0 route among 96 at `3059c55f`. The closed gripper approached the rear rim with tilt 28°, yaw 180° and flank −35°. The plate entered the native goal at control 204 with reward one. The separate positive camera collector `job-p0ct7jplcisf` found eight successful Sawyer episodes in 96 attempts. The larger source banks use fresh normal-camera resets, positive terminal reward and genuine issued commands. [Native collection](sweep_bank.py)

| Work | Baige distributed job | Code |
| --- | --- | --- |
| Panda official replay and feedback approach | `job-cb3620vqbqoa` | `9c1225b9` |
| Sawyer headless source search | `job-hobqjf487g5p` | `3059c55f` |
| Sawyer direct camera search | `job-p0ct7jplcisf` | `3059c55f` |
| Wider camera-verified Sawyer bank | `job-4nfdkzgstd6y` | `6d901fb2` |
| Wider camera-verified Panda bank | `job-q2cl6gz44q8a` | `6d901fb2` |
| Two-arm PaliGemma source SFT | `job-1pd910mqu8pe` | `e22c1d88` |
| Additional source-only Sawyer seeds | `job-n61xfw8ba6ll` | `6d901fb2` |
| Additional source-only Panda seeds | `job-e9vmrtk43yya` | `6d901fb2` |

Each allocation requested one NVIDIA RPBZZZ6, 12 CPUs, 96 GiB host memory and 95 minutes. The first two-arm SFT snapshotted 45, 8, 41 and 58 verified camera trajectories in the four input banks. Deduplication yielded 81 Panda training tapes from 28 layouts and 59 Sawyer training tapes from 27 layouts, plus five offline tapes on four separate layouts for each arm. This first offline bank includes positive trajectories obtained by a search fixed before collection. Fresh closed-loop development and confirmation stay separate. At 13:02 Shanghai the optimizer had advanced through update 1,175 of 1,600, at approximately 0.12 s per batch of eight. [Training](motor_train.py) · [Gates](motor_eval.py)

The first fit trains low-rank attention weights in the last two Gemma layers and a fresh action head. Expanded comparisons retain allocated-renderer evidence per episode and exclude validation labels from per-layout parameter search. Shared run root is `/mowenjie/fluffy-bassoon-2/outputs/libero-paligemma-dualarm-20261120-r2`.

## Expanded qualification

Both extended 6,400-update source fits completed on Baige. The 1,600-update original checkpoint and the 6,400-update regression and categorical checkpoints were registered for the same 50 fresh native samples per arm, before scoring. Source admissions reapply the actual Panda donor state on both arms. Each candidate receives 100 closed-loop episodes. The held-out test has a separate seed range. [Sampling and decision rules](PLAN.md) · [Evaluation](LARGE-PANEL.md)

| Work | Baige distributed job | Evidence location under the shared run |
| --- | --- | --- |
| Extended regression SFT | `job-1wu7hwcwz4pb` | `sft02/fit` |
| Extended categorical action SFT | `job-8chpqr0tr9kp` | `sft03/fit` |
| Source recovery | `job-lf7wfg7kuxrh` | `recover01/bank/index.json`, 49 positive suffixes from 120 handoffs (17 Panda and 32 Sawyer), 3,653 positive teacher controls. The bank completed before the launcher reported its nonzero pre-existing threshold exit. |
| Counterfactual replanning frequency on the four pilot layouts | `job-ybo867hh45g6` | `stride01/dev/report.json` |
| Successful trajectory auxiliary labels | `job-qdo245oqvkdu` | `labels01/bank/index.json` |
| Waypoint-supervised source fit | `job-6un3g38a249n` | `waypoint03`. The preceding fixed-demo reader expected a trace absent from the published Panda recording. Replay now checks every recorded wrist state, issued action, measured motion and native reward. One fixed Panda validation trajectory passed. |
| New native training BDDL draws | `job-isezj1jp5rg6` | `gendemos01/banks`, first 96 distinct admitted placements sampled within the separate 300000–300127 training window, up to 16 route searches per arm and layout |
| Original checkpoint on the 50-layout source panel | `job-w4shd13nxfin` | `bigbase02`, `large-panel-v2/dev` |
| Regression checkpoint on the 50-layout source panel | `job-jkzwhp5e2ljj` | `bighuber02`, same panel |
| Categorical checkpoint on the 50-layout source panel | `job-lpvjnb88lzlo` | `bigcat02`, same panel |

The first preparation generated 50/50 distinct layouts on seeds 740000–740049. The initial scoring jobs exited before any episode because the evaluator revision had advanced after preparation. The replacement preparation and scoring use the same immutable `e2f794c4` source. Keep the original preparation and failed launcher receipts in their original directories.
