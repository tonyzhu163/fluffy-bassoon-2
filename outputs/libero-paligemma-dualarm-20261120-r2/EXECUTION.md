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
