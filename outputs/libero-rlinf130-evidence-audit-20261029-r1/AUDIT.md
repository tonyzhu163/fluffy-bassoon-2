# Independent completed RLinf corpus/development audit

Read-only snapshot of four completed Baige runs. Every run exited0 and its
final summary says completed. No model inference/training or executable pickle
deserialization was used for this audit; only JSON, SHA-256 and NPZ member bytes.

| Evidence | Source pin | Episodes | Successes | Primitive actions | Settling/qualification actions | Worker seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Original clean source | `d6a981037b424eb27a38c06ce5c6a240818954e5` | 24 | 24 | 3,371 | 250 | 441.59 |
| Additional Plus source | `42fd5176f484aa6b884e52881a2a3021ff18e258` | 24 | 21 | 3,692 | 260 | 603.58 |
| Plus development, all five baselines | `b3d1df4a9b77c66650c011e1e70c17f8cdb6a8f1` | 40 | Method-dependent | 6,373 | 420 | 604.51 |
| Pro development, all five baselines | same | 40 | Method-dependent | 6,940 | 210 | 627.68 |

Total audited cost:128 episodes,20,376 executed primitive actions +1,140
settling/qualification actions =21,516 simulator actions. Worker elapsed time
2,277.37s; scheduler/setup/test time is additional, not inferred as zero.

## Source evidence

- Exact shared released motor revision `6222623f635769bfc73c9472e29fab9b7fd8e027`;
  strict667-tensor loading, no missing/unexpected tensors. Receipt and feature
  contracts are retained in each raw contract/index.
- Clean: four training scenes0–3,16 episodes,2,243 primitive steps,4,600 SAC
  updates; two local-validation scenes36–37,8 episodes,1,128 steps,2,320 updates.
  Six distinct complete trials;24/24 source episodes succeeded.
- Plus source: training cases2387(light) and1438(sensor noise), three scenes
  each, all train-only.21/24 successes;3,692 primitive steps,7,560 updates.
  Existing development cases667/290 and all Pro cases are excluded.
- All source trials: fresh full common SAC initialization, K4, batch32,
  UTD20, warmup0; same fixed4104D RLinf features.
- All six Plus source first attempts succeed. Success totals by attempt:
  6/6,4/6,5/6,6/6. The lighting trial sequences are
  `[true,false,true,true]`, `[true,true,false,true]`,
  `[true,false,true,true]`; sensor-noise trials succeed throughout.
  No monotonic teacher improvement is established.
- Final deterministic actors supply candidate DPT-style labels. They have no
  independent expert-competence evaluation. Behavior-policy source success
  does not supply that missing result.
- Clean-only unique collection:3,371+250=3,621 physical actions. Augmented
  unique collection:7,063+510=7,573 actions, including the same clean validation.
  Paired objectives reuse each source interaction; do not double-count them.

## Development evidence

| Method | Plus successes | Plus primitive steps | Pro successes | Pro primitive steps |
| --- | --- | --- | --- | --- |
| Native | 8/8 | 1,034 | 8/8 | 1,159 |
| Zero | 8/8 | 1,000 | 8/8 | 1,068 |
| Tied Gaussian | 4/8 | 1,865 | 4/8 | 1,817 |
| Zero-first search | 8/8 | 1,000 | 8/8 | 1,068 |
| Fresh DSRL online | 6/8 | 1,474 | 4/8 | 1,828 |

- Two variants per suite, one nominal scene each, K4 correlated attempts.
  Original object task1, cream cheese→basket, maps to training goal family47.
- Search starts with zero and retains success; every search first attempt
  succeeded. Equal aggregate steps with zero do not imply a search benefit.
- DSRL online: fresh policy, batch32/UTD20/warmup32;2,040 updates on Plus,
  2,560 on Pro. Both cases succeed on attempt4, but native succeeds throughout.
- These are development cases, not final benchmark scores. No frozen
  history/static student was evaluated in these four audited runs.

## Checks and reproducibility

- [remote-verification.json](remote-verification.json):290 file fingerprints,
  including168 hashes matched to previously saved dataset/trace/checkpoint
  receipts. Large traces and checkpoints remain on Baige.
- Across all12 source trials: every AD/DPT NPZ member except query targets is
  byte-identical; AD targets equal executed noise. Dataset identities and
  per-trial complete boundaries retained. No model-based relabeling was run.
- [derive.py](derive.py) recomputes [RESULTS.json](RESULTS.json) from the
  compact [raw mirror](raw/), checks every mirrored fingerprint and cross-checks
  counts/costs against final indices/summaries. [verify_remote.py](verify_remote.py)
  records the read-only remote binary-integrity/paired-context checks.
- Focused allocated tests: clean102 passed; Plus source107 passed; Plus/Pro
  development97 passed each. Raw final test summaries are in the verification file.

Interpretation: one-goal-family collection/deployment smoke test. Clean success
ceilings, non-monotonic Plus source behavior, and native development ceilings
provide weak evidence for a useful learned adaptation process. They establish
neither new-goal generalization nor expert final candidates nor a memory benefit.
Adding Plus training changes Plus from unseen-benchmark transfer to within-Plus
held-out-case generalization; Pro remains absent from attachment pretraining.
