# Source tasks, reader refits and prediction export

The original tuned Perceiver and seven-shift SymMeta both trained on all 24 source task IDs. Their saved checkpoints therefore cannot establish performance on those tasks as unseen. Both readers are refitted from initialization on the same 12-task partition with their respective published training recipes. This screen tests transfer across source task IDs under seven physically executed interfaces. The shift values are familiar to the seven-shift reader. No official development or test tape enters any stage.

## Registered task and support grid

The partition was recorded in the shared worker contract before holdout predictions or errors were computed. Each suite contributes three reader-fit tasks, one calibration task and two evaluation tasks. The earlier eight-task source collection provides independent judge-fit recordings. Its tasks belong to the reader-fit partition. Its seeds, source archive and recorded traces differ from those of both reader fits.

| Suite | Reader fit | Independent judge fit | Gate calibration | Evaluation |
| --- | --- | --- | --- | --- |
| libero_10 | 1, 2, 9 | 2, 9 | 6 | 4, 7 |
| libero_goal | 2, 5, 7 | 5, 7 | 8 | 4, 9 |
| libero_object | 2, 3, 6 | 2, 6 | 5 | 4, 9 |
| libero_spatial | 2, 3, 4 | 2, 4 | 6 | 5, 9 |

Calibration and evaluation each cross every listed task with Panda and Sawyer, starts 0–3, and the seven actual interventions `nom`, `yaw90`, `yaw180`, `yawm90`, `scale_lo`, `scale_hi`, `grip_inv`. This gives 224 calibration and 448 evaluation recordings. Each contributes exactly three chronological five-control chunks and four reader views for matched seeds 41–43. The 2,016 trace–seed rows include 1,344 evaluation rows. The frozen π₀.₅ policy generated every passive support in MuJoCo. The matched starts within arm and task share their initial physics and opening policy noise across shifts. The seven shifted rollouts have separate executed controls and observed motions. Each physical tape appears once before crossing reader seeds. The test contains no artificial interface relabels. [Scale collector](../libero-xemb-ur5e-scale-20261120-r1/collect_scale.py) · [Source task definitions](../libero-meta-learning-20261117-r1/SPLITS.json)

## Archives and separation

All paths below are relative to `/mowenjie/fluffy-bassoon-2/outputs` on Baige.

| Role | Recording origin | Selected data |
| --- | --- | --- |
| Tuned Perceiver fit | `xemb-icml-campaign-20261121-r1/gpu12scale/stage3/diverse_shards` | 732 of 1,466 source trajectories, 426 Panda and 306 Sawyer, covering all 128 actual source interfaces. PR7's original lexical shard order and on-minibatch continuous family relabels apply. |
| SymMeta fit | `libero-xemb-ur5e-scale-20261120-r1/gpu01/archive` | 672 of 1,344 original source tapes, both arms, starts 0–3, all seven shifts. The original archived order is retained. |
| Judge fit | `libero-xemb-ur5e-sysid-20261120-r1/gpu02col/archive` | 224 independent passive tapes, eight tasks × two arms × two starts × seven shifts. Starts are 0 and 1 under the earlier collection's seeds. |
| Gate calibration and held-out screen | Same scale archive as the SymMeta fit | Entirely separate task IDs, respectively 224 and 448 tapes. The public scorer inputs contain chunk RGB pairs, issued command XY, vertical displacement, validity and identifiers. True factors for each held-out role travel in a separate file. |

The archived 320-row dual-arm pilot has 224 source rows on the earlier eight training tasks and 96 development rows on UR5e. The preceding single-arm pilot has three policy supports off the training split, all on Goal 1. The dual-arm Goal 5 controller collections repeat a reader-training task. The multi-variant camera collector uses probes on the original 24 training tasks. The stage-3 archive includes more physical shifts on these same 24 task IDs and stores three valid chunk slots. The scale archive additionally records each support's actual control count, which establishes the exact exposure for the registered balanced screen. [Earlier dual-arm collector](../libero-xemb-ur5e-sysid-20261120-r1/collect_ur5e.py) · [Single-arm collector](../libero-xemb-sysid-20261119-r1/collect.py) · [Stage-3 collection plan and custom simulator loop](../xemb-icml-campaign-20261121-r1/eval_stage3_scaling.py) · [Multi-variant plan](../xemb-icml-campaign-20261121-r1/eval_pro2_visual_stress.py)

The scale archive index has SHA-256 `37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076` and collector pin `d597ee0f1e99937bbeb2705132d709b514a4b5c2`. The pilot index has SHA-256 `65c079f1051e347d781d950643113be9d4de6b92e952e373503d78d0eeb3b82e` and collector pin `d639787a1eb0c0cf41b8c2557ea39ed0d2f282a3`. The stage-3 collector pin is `c599c861f7d76e7b0b17d70520b47089671728ee`. Input preparation verifies all eight stage-3 NPZ hashes, metadata order, shift factors and original validity masks against the published source lock and collection plan. It also verifies all 1,344 scale tape hashes against the earlier independent source-membership export. Each pilot and scale tape must have the registered arm model, three complete chunks, adjacent frame equality and exact reset image/state digests. The new membership records retain simulator seeds, starts, model classes, physics fingerprints, noise and opening-proposal digests. The source seal records hashes of the published arrays and memberships. [PR7 archive lock](../symmeta-perceiver-retry-20261123-r1/export-inputs.json) · [Prior scale membership](../symmeta-matched-flow-20261123-r1/evidence/source-membership.json.gz)

The judge-fit file additionally includes the nominal transformed command and world-frame observed displacement. The former is the ideal labeled rotation/gain applied to the issued chunk mean, before any per-control actuator clipping. The latter rotates the recorded local wrist displacement back into world coordinates. Calibration/evaluation readers receive an explicit seven-field sensor whitelist and no true factors. The scorer receives evaluation labels only during its final scoring phase. [Scorer registration](SCORE.md)

## Reader handling and qualification

The tuned reader calls the original PR7 `train_model` on the restricted input, with 24 learned position slots, batch 32, 2,500 updates, a peak learning rate of 0.0003 and 200 warmup updates. The geometric reader uses the Stage 1 `equiv_v2_dis_dz` configuration with 32 slots, batch 64 and 3,500 updates under cosine decay from 0.001. Both use their original seed-local samplers, optimizer, DIS representation and factor loss. On Baige CPU, one 64-example update of the geometric training loop matched the original Stage 1 trainer bit-for-bit on all 33 parameter leaves. Historical full-data weights are used only to qualify decoding on reader-fit tapes. Final predictions require the freshly refitted checkpoints and their source and revision seals. [Original tuned trainer](../symmeta-wider-family-20261123-r1/eval_wider_family.py) · [Stage 1 trainer](../xemb-icml-campaign-20261121-r1/stage1_offline_gate.py) · [Published export](../symmeta-perceiver-retry-20261123-r1/export.py)

For each held-out slot, the reader zeros its RGB, flow, command and proprioceptive arrays and clears its validity bit. The other two slots keep their original contents, chronology and learned position indices. The published decoder and its full-precision NumPy yaw, gain and sign handling are retained. SymMeta runs per case and the tuned reader runs in batches of 96. Each prediction carries an angular-validity bit for nonzero returned cosine/sine and any available support. All combined candidates use the three-chunk tuned gain and gripper sign.

Adjacent chunks share their boundary camera frame. Omitting the middle token leaves both of its endpoint frames, separately, in its two neighbors. The held-out command, paired flow and measured displacement are excluded as token inputs. This design measures omitted-chunk consistency within one contiguous trace. The three votes are correlated.

CPU qualification on the full source inputs found 224 distinct three-command sequences among 224 judge-fit recordings, 224 among 224 calibration recordings and 448 among 448 evaluation recordings. It tested four real fit tapes with both historical seed-41 checkpoints. Both full decoders matched exactly. Poisoning the omitted RGB, flow, action and proprioception changed zero predictions. Compacting to two tokens and restoring each kept token's original learned position reproduced every masked yaw, gain and sign exactly. All 24 reader–tape–omission yaws differed from their full predictions. Four views of 32 traces took 0.375 seconds for the tuned reader and 0.373 for the geometric reader using two CPU cores and batch 32. The first diagnostic reader invocation stopped during parsing. The corrected fresh attempt completed before any held-out prediction. Production repeats the mask tests for each refit. [CPU result](reader-evidence/development-cpu.json) · [Development source receipt](reader-evidence/development-source.json)

## Reproduction

Run from the clean detached parent-deployed checkout. Provide its full revision in `PIN`. Source packaging, qualification and inference use two allowed Linux CPU cores. The parent submits the refit command in an exclusive Baige Distributed Training container with one RTX 4090 or RTX PRO 6000D and a 90-minute job limit. Each command creates a fresh attempt directory.

```bash
run=outputs/symmeta-omitted-chunk-20261124-r1
bash "$run/run_readers.sh" data "$PIN" data01
bash "$run/run_readers.sh" qualify "$PIN" qual01 data01
# Parent's distributed job
bash "$run/run_readers.sh" refit "$PIN" reader01 data01
# Baige CPU after successful refit
bash "$run/run_readers.sh" predict "$PIN" pred01 data01 reader01
```

The shared output root is `/mowenjie/fluffy-bassoon-2/outputs/symmeta-omitted-chunk-20261124-r1`. `data01` contains the sealed source archives, `reader01` contains the six restored checkpoints and `pred01/readers.npz` carries full and omitted yaw predictions. `pred01/prediction-receipt.json` authenticates the prediction file and its dependencies. Repeats must use new attempt names. [Source builder](source.py) · [Reader driver](readers.py) · [Launcher](run_readers.sh)
