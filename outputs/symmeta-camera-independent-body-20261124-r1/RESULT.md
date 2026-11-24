# Kinova3 physical replication

**Explicit registration reduced yaw error on the new physical Kinova3 supports.** With all three source-trained seed pairs, A averaged 12.03° and the fully informed spatial Perceiver B averaged 15.41°. The paired reduction was 3.38° (95% whole-task interval 0.72° to 6.48°). The shipped PR7 with the same public warp averaged 12.50°. Its paired difference from A was small relative to uncertainty. A had slightly larger gain error than B. All four methods recovered every gripper sign. Fresh native success remains unmeasured in this release.

## Locked comparison

The parent pinned the [plan](PLAN.md) and [64-case schedule](cases.csv) at `69f44b996b4c22ba5ffa3133f0079fa671c57bc8` before discovery scores were available to this workstream. The schedule uses all eight original BASE tasks, four continuous and four compositional OOD support parents per task and sign-balanced ±36° physical orbits. It fixes one challenged observation per parent, layouts 26/27 for queries and reader seed 41 for eventual native deployment. All arms receive the same cases, resets and noise. The trained readers use the primary source-selected update 5000 from Panda/Sawyer. No Kinova3 examples entered model fitting or checkpoint selection. The released predictions include all three source seeds and their full heads. [Primary source seal](evidence/score01/primary-seal.json) · [Full predictions](evidence/score01/predictions.json.gz)

| Frozen reader on assigned physical view | Yaw MAE | Gain MAE | Gripper accuracy |
|---|---:|---:|---:|
| A, explicit registration | 12.03° | 0.0921 | 100% |
| B, spatially conditioned Perceiver | 15.41° | 0.0833 | 100% |
| Shipped PR7 with identical public warp | 12.50° | 0.0904 | 100% |
| Shipped PR7 on observed RGB | 32.16° | 0.0926 | 100% |

For A minus B, the gain MAE difference was +0.00874 (95% interval +0.00392 to +0.01312). For A minus shipped+warp, the yaw MAE difference was −0.47° (−1.65° to +0.88°). Swapping H to the opposite calibration on the same observed pixels raised yaw error by 56.01° for A and 32.25° for B. Six of eight tasks favored A over B on yaw. Effects ranged from a reduction of 11.94° on libero_object/8 to an increase of 1.19° on libero_spatial/8. Each interval resamples whole tasks together with all matched cases and seeds. Across these eight tasks, the correlation between A and B task-mean yaw errors was 0.53. This descriptive association and the paired intervals are conditional on the eight tasks. [Per-task, seed, sign, regime and paired estimates](evidence/score01/offline.json)

These controls support using public registration for yaw inference on the second arm. They leave the additional yaw benefit of refitting over ordinary preprocessing unresolved. The simultaneous gain disadvantage matters for the native question. The frozen seed-41 yaw MAEs for A, B, shipped+warp and raw shipped were 11.74°, 14.71°, 14.29° and 33.98°, respectively. Their released corrections include their own challenged-view gain and sign. Only matched new episodes can establish whether the full correction changes native success. The privileged true map is reserved as a separate fifth arm.

## Physical qualification

All 64 distinct archive parents yielded a fresh seven-joint Kinova3 trajectory with three camera views and four native observation boundaries per view. All 192 NPZ identities are distinct. The sensor hook restored the camera, checked equal integration state across renders and matched the nominal pixels and TCP to the same observation clock. All 576 independently rasterized landmark projections were within 0.0874 pooled pixel. The first physical fixture's maximum error was 0.0517 pixel. All 16 task/layout query resets completed at step zero on the registered arm. [Bank and per-view hashes](evidence/bank64.json.gz) · [All trajectories, issued and executed commands, projections and clock diagnostics](evidence/diagnostics.tar.gz) · [Query reset records](evidence/cpu01/query-resets.json)

The original π₀.₅ passive bank contains chunk means. The new render executed each archived mean for five controls. None of the 64 fresh whole-state fingerprints equaled the archived fingerprint. Across cases, the largest difference in individual chunk joint displacements ranged from 0.00052 to 0.01758 rad. Every view and reader used the new trajectory's own proprioception. All eight OOD cases from the previously discrepant libero_spatial/8 reset were retained. The evidence supports same-trajectory camera pairing. Exact reconstruction of the historical closed-loop motion remains unverified. [First fixture and source identities](evidence/Kinova3-0008/row.json) · [First physical diagnostics](evidence/Kinova3-0008/diagnostics.json)

The one-case reader fixture verified identical observed RGB, H and sensor hashes for all four readers, identical RGB inputs to A and shipped+warp, and identical RGB inputs to B and raw shipped. The interface fixture reconciled the archive's means with fifteen executed controls and tested the independent inverse/forward command calculation at two full corrections. Neither fixture stepped a native query. Rendering, reset qualification and both one-case checks used 796 seconds of CPU qualification wall time in total on cores 36/37 (13 min 30 s from first start to last exit, including handovers). All three stages exited zero. [Feature readback](evidence/fixture01/fixture-inputs.json) · [Native interface readback](evidence/interface01/native-interface.json)

## Independent release

The parent ran the full offline inference in Baige Distributed job `job-m8k5ugoqkbss` on one RTX 4090 and released 1,344 prediction records. The 64 native records each contain A, B, shipped+warp and raw shipped seed-41 corrections plus the separate true map. No native outcomes or historical Kinova successes were used in this comparison. The native launcher records per-step rewards, emitted actions, opening proposals, policy-noise fingerprints and complete matched five-arm sets. Its first real GPU episode remains to qualify. [Full-head release](evidence/score01/full-head-release.json) · [Parent launch contract](HANDOFF.md) · [GPU receipt](evidence/score01/gpu.json)

| Parent release input | SHA-256 |
|---|---|
| Shared `cpu01/bank64.json` | `8639817ce59b79ecfe26564e94c0bb61352a3a5942b13a5c8edd999a6c5d6e60` |
| Shared `cpu01/query-resets.json` | `cb4309ddea85347bf06798cfd8156222a827f94bc8a0d06153ed157f2fa69ec4` |
| Primary `fit01/sealed.json` | `b9e16fe2ff43fac141be0590beed58fe073dbf25a46ed214efbfd8f6068cd16e` |
| Shared `score01/full-head-release.json` | `50e320040e06c70d2b0924f14afc24f0608e5951bfc7bcec37f5781384c7820f` |

Shared files are under `/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-independent-body-20261124-r1`. Parent native checkout `3a6f90a25fbb6b2de0b06b02a9c11aff1f34b3f3` contains the pinned launcher, full-head verifier, analysis and single-case fixture. The parent owns submission and the remaining share of the 2.5 additional RTX 4090 GPU-hour envelope.

The first native job stopped before its first episode because the GPU probe received no device index. The retry explicitly sets the one allocated device, uses the original full-factor predictions and reset hashes, and writes to a fresh directory. Baige assigned job-nn97ohkc3wf8 to the retry. [First probe](evidence/native01/gpu.json)
