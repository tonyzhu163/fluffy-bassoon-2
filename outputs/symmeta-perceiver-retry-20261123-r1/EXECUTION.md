# Four-panel native retry on Baige

At 16:23 Shanghai on November 23, Baige accepted `job-am9y13b811bk` for four RTX PRO 6000D GPUs on one worker. The requested wall time is four hours. The two readers will each run 1,152 episodes, paired on support tape, reader seed, query reset and initial policy proposal. The scoring program uses the native simulator success flag. The first check after dispatch verifies GPU binding and a saved episode pair. [Launch receipt](evidence/native-job.jpg) · [Runtime and comparison keys](LIVE.md)

## Frozen input

Each released panel contains the same 96 scheduled command settings and tasks, including 64 OOD cases, with its own passive support tapes. The checkpoint and prediction choice was made before collecting a native outcome. The RTX 4090 refit on the original accelerator, `job-6uezn9kt01vj`, was waiting for capacity. The first complete refit, Pro6K job `job-dg2hu5uswgea`, ran the three registered seeds on the same PR7 recipe. Each 24-position Perceiver trained for 2,500 updates at batch 32. Each saved checkpoint reproduced its 96 pre-save correction triples after reloading. [Refit recipe](TRAINING.md) · [Checkpoint digests](evidence/train02pro/checkpoints.sha256)

| Reader seed | Published RTX 4090 OOD yaw MAE | Pro6K refit OOD yaw MAE |
| --- | ---: | ---: |
| 41 | 5.42° | 5.64° |
| 42 | 5.83° | 5.98° |
| 43 | 5.48° | 5.84° |

These columns use the same 64 recorded UR5e supports and PR7 scorer. Native rollouts use the Pro6K refit's full correction triples, including scale and gripper sign, and the original same-seed SymMeta checkpoints. The refit took 274 launcher seconds (0.076 GPU-hours). [All per-seed score differences](evidence/train02pro/training-sanity.json) · [GPU receipt](evidence/train02pro/gpu.json)

The training revision was `ba3c41e25575d07d7bd8bbfaae2797e7cf9dc0a9`. The native evaluator revision is `d568535b758f2b97b382f9b3e280376e769b5942`. CPU preparation verified all 384 tape SHA-256 digests and joined the 1,152 prediction rows to the four native query schedules. The immutable predictions have SHA-256 `c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b`. [Frozen predictions](evidence/train02pro/predictions.json) · [CPU preparation](evidence/prepare01pro/contract.json)

Full checkpoint files and raw episode records are kept under `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-perceiver-retry-20261123-r1/`. The refit directory is `train02pro/`, the verified input is `prepare01pro/`, and the submitted native attempt is `native01pro/`.
