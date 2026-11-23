# Four-panel native retry on Baige

**Complete, November 23, 19:14:29 Shanghai.** Baige Distributed Training job `job-2s6q9a9xe96e` exited 0 on four RTX 4090s. Attempt `native02g4090` completed all 32 panel–task shards with 1,152 fresh pairs, including 768 OOD pairs. Independent CPU readback verified both resets, opening proposals and native reward traces for every pair and reproduced the official summary. The launcher ran for 8,543 seconds (9.492 GPU-hours). Including both reader refits, the run used 9.666 launcher GPU-hours. [Four-panel results](RESULT.md) · [Exit receipt](native02g4090/exit.txt) · [Baige completion receipt](evidence/native-complete.jpg) · [Refit exits on Pro6K](evidence/train02pro/exit.txt) and [RTX 4090](evidence/train01/exit.txt) · [Independent readback](evidence/native02g4090/readback.json)

## Allocation history

At 16:23 Shanghai on November 23, Baige accepted `job-am9y13b811bk` for four RTX PRO 6000D GPUs on one worker with a four-hour wall limit. The plan called for 1,152 episodes per reader, paired on support tape, reader seed, query reset and initial policy proposal, with native simulator success as the outcome. [Launch receipt](evidence/native-job.jpg) · [Runtime and comparison keys](LIVE.md)

The first allocation remained queued because no node had four free GPUs. At 16:33 Shanghai I stopped that specific unstarted job and submitted four single-GPU Distributed Training jobs. Each uses the frozen evaluator, predictions and eight tasks from one panel. The scheduling wrapper is `14dab6c95067fa208a4047951375de2afcb4774b`. In a Linux CPU qualification, the four preparations reproduced the sealed inputs and the combiner preserved every pair in a 1,152-pair fixture. [Panel launcher](fragment/panel.sh) · [Strict combination](fragment/fragment.py) · [Submitted jobs](evidence/native-panel-jobs.jpg)

| Unstarted Pro6K fallback | Baige job | Assigned pairs |
| --- | --- | ---: |
| UR5e × π₀.₅ | `job-60xi82krb40p` | 288 |
| IIWA × π₀.₅ | `job-tvbb9yhfqy4z` | 288 |
| Kinova3 × π₀.₅ | `job-sg069yja48p2` | 288 |
| UR5e × π₀ | `job-0jaby7lttjel` | 288 |

At 16:51 Shanghai the Pro6K queue reported zero free GPUs, and all four panel jobs were still queued. The RTX 4090 queue had a schedulable four-card node. Baige started `job-2s6q9a9xe96e` on that node at 16:52. The launch passed the same frozen Pro6K predictions to the same `d568535b` evaluator. Each pair used one physical GPU for both readers. All four pending Pro6K fallback jobs were stopped before dispatch and wrote no attempt artifacts. The completed attempt is `native02g4090/`. [Allocation and released reservations](evidence/native-4090.jpg)

At 17:05 the four-device probe and inference/rendering bindings had passed. The first 14 UR5e pairs were present. Case 0, seed 41, had 520 controls per reader, identical full resets and identical opening proposals under NVIDIA EGL.

## Frozen input

Each released panel contains the same 96 scheduled command settings and tasks, including 64 OOD cases, with its own passive support tapes. The checkpoint and prediction choice was made before collecting a native outcome. The RTX 4090 refit on the original accelerator, `job-6uezn9kt01vj`, was waiting for capacity. The first complete refit, Pro6K job `job-dg2hu5uswgea`, ran the three registered seeds on the same PR7 recipe. Each 24-position Perceiver trained for 2,500 updates at batch 32. Each saved checkpoint reproduced its 96 pre-save correction triples after reloading. [Refit recipe](TRAINING.md) · [Checkpoint digests](evidence/train02pro/checkpoints.sha256)

| Reader seed | Published RTX 4090 OOD yaw MAE | Pro6K refit used for native retries | Fresh RTX 4090 refit |
| --- | ---: | ---: | ---: |
| 41 | 5.42° | 5.64° | 5.12° |
| 42 | 5.83° | 5.98° | 5.05° |
| 43 | 5.48° | 5.84° | 5.56° |

These columns use the same 64 recorded UR5e supports and PR7 scorer. Native rollouts use the Pro6K refit's full correction triples, including scale and gripper sign, and the original same-seed SymMeta checkpoints. The refit took 274 launcher seconds (0.076 GPU-hours). [All per-seed score differences](evidence/train02pro/training-sanity.json) · [GPU receipt](evidence/train02pro/gpu.json)

The original-accelerator refit subsequently completed in 350 launcher seconds (0.097 GPU-hours). Its three restored checkpoints each reproduced all 96 correction triples from memory. Across the 288 seed–support combinations in each panel, the two refits differed by 3.37° to 5.44° in mean absolute predicted yaw. They agreed on 1,150 of 1,152 gripper signs. Aggregate yaw accuracy reproduced across refits, with measurable differences in individual corrections. The frozen native input remains the Pro6K prediction file. [RTX 4090 per-seed comparison](evidence/train01/training-sanity.json) · [Export and checkpoint hashes](evidence/train01/hashes.txt)

The training revision was `ba3c41e25575d07d7bd8bbfaae2797e7cf9dc0a9`. The native evaluator revision is `d568535b758f2b97b382f9b3e280376e769b5942`. CPU preparation verified all 384 tape SHA-256 digests and joined the 1,152 prediction rows to the four native query schedules. The immutable predictions have SHA-256 `c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b`. [Frozen predictions](evidence/train02pro/predictions.json) · [CPU preparation](evidence/prepare01pro/contract.json)

Full checkpoint files and raw episode records are kept under `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-perceiver-retry-20261123-r1/`. The refit directory is `train02pro/`, the verified input is `prepare01pro/`, and the completed native episode records are under `native02g4090/pairs/`. [Official summary](native02g4090/summary.json) · [Independent compressed pair records](evidence/native02g4090/pairs.csv.gz)
