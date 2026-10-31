# LIBERO bounded baseline completion — 2026-10-31 amendment

This amends the LIBERO qualification/status/performance rows from pending to
completed ONLY where the committed execution-owner receipt supports it.
Receipt commit: `1a65226e0d3b6a68cf59808028245b3da5be3d07`,
`outputs/libero40-baseline-completion-20261030-r1/{README.md,receipt.json}`
in `/Users/mowenjie/.codex/worktrees/vla-performance-pilot/fluffy-bassoon-2`.

Both files were read back and matched byte-for-byte to that commit. Receipt
SHA256 is `d8f1f4277e5600e0e27db242c8909055170ba9b3894a35bedd921e657b285598`;
README SHA256 is `c9daf1b106306b2fa61cc969b80c45bc59892582b84aecf061ea94399240736b`.
Its saved-metadata verification timestamp is2026-10-30 23:57:50 Shanghai.
This amendment performs JSON/hash/arithmetic checks only; it does not rerun
models/tests, rehash changing active checkpoints, allocate resources or alter
any budget. `LIBERO-BASELINE-PROVENANCE-20261031.json` preserves the thin handoff.

## Completed components and their own controls

| Component | Completed result | Interpretation limit |
| --- | --- | --- |
| Frozen PPO source validation |32 K4 trials /128episodes; full63/64, separately trained reset61/64. First two attempt positions are16/16 for each |Source-local, near ceiling; not all-target performance or memory benefit |
| Full90 noise controls |Native29/90; zero25/90; tied Gaussian28/90; tied clipped29/90 |Use THIS native29 control for these conditions |
| Frozen PPO full90 K0 targets |Native30/90; frozen full24/90; independently trained reset26/90 |Panel COMPLETE. Learned policies are below their paired native control; no target fitting |
| Fixed-nine PPO K1 panel |Full5/9; SAME-WEIGHT episode reset5/9; independently trained reset5/9 |No observed difference in this small one-support/query panel; it is not K8 or multi-seed evidence |
| Source SAC competence |Native16/16, deterministic mean16/16, sample16/16 |EIGHT task-specific source actors,48 traces; not a joint all40 SAC or all40 competence claim. No source teacher advantage measured at this ceiling |
| Six-case target-online PixelSAC onset |4/6 queries at each training-episode budget0/1/4/8/13/16;96 train +36query episodes |Completed EARLY ONSET screen, not convergence. PixelSAC image64/proprio8 differs from history prefix4104D |

These are the fixed-bank, one-seed/small-panel results in the owner's receipt.
Do not promote them to an independent multi-seed estimate, reliable memory
advantage, sample-efficiency gain or robust ICRL/architecture result. Existing
AD/DPT fit/model pins remain binding. This receipt adds completed CONTROLS;
it supplies no new AD/DPT target score and does not implement separately
trained supervised episode-reset, GRU or CTM cells.

PPO target completion covers full90 at K0 plus the fixed-nine K1 panel; it
does not fill the entire planned0/1/2/4/8 support grid. The six-case PixelSAC
0/1/4/8/13/16 values are online training-episode budgets, a different axis from
history-policy support counts.

Keep all three native comparisons separate:

| Native panel | Result | Correct use |
| --- | --- | --- |
| Earlier student firstwave (`d095a2a4e66a0b79d03c9e823a183c19452d0c10`) |31/90 |Comparator for that student wave only |
| Noise controls |29/90 |Comparator for zero/tied/clipped noise only |
| Frozen PPO target wave |30/90 |Comparator for frozen full/trained-reset PPO only |

Do not pool31/29/30 or substitute controls across waves. The saved31-versus29
check matched all90 initial/seed/parameter metadata; four cases had outcome
flips despite identical first input/noise arrays and differing decoded first
actions. Cross-process decoder sensitivity was observed; no particular kernel
cause was established. This does not turn the three runs into interchangeable
paired references.

## Source-only model and execution identities

Frozen PPO model SHAs are unchanged across source validation and BOTH target
waves, as checked in the committed receipt:

- Full: `9d8d5a08ef2a30abbef579ed810d3e6ae7c998cb576909f0c6d5c0e98889e904`.
- Independently trained reset: `15916bf91ec15adf8ffb3d6f36b957f5e263c96b2c5c7b96e3ada58c71d010d1`.
- Original full source recovery: `c2f15e0918c3f211e1629416fa572027ef103510365d42b0149a83b08f3bc214`.
- Original reset source recovery: `babe7cf68ab5efe572448629d81dea5ba4974116f2361f7f19a0b9d3eb32fcc7`.

The original recovery and eight task-specific SAC source actor checkpoint
hashes are preserved in the JSON handoff directly from the committed receipt.
Model selection stays source-only. No target outcome selected these frozen models.

| Component | Exact execution code | Baige jobs | Actions | Charged GPU-seconds |
| --- | --- | --- | ---: | ---: |
| Source PPO validation |`9002a2dd352e6176c43943c1e46bf2b85eddc397` |`job-ztzwinm2b802` |22,753 |1,203 |
| Full90 noise controls |`53b6c0d1bcccd6d94025762335b66a72f09a3119` |`job-g8vas173mmy5`, `job-834arjyjj6mp` |122,993 |4,411 |
| Frozen PPO target wave |`7780fb38b61fb37a3e2c0aec67ffea8e8e9c0b94` |`job-3ckrfuvifjwk`, `job-mrfdf4ipqicm` |108,894 |4,526 |
| Failed SAC setup R1 |`7780fb38b61fb37a3e2c0aec67ffea8e8e9c0b94` |`job-qioinn358m7l` |0 |78 |
| Source SAC competence R2 |`492c289e2ecb56c6e38034097aaab69746b0fd8c` |`job-zxgkfebhs4k8` |7,540 |532 |
| Six-case target-online onset |`ba4674fafbe6d8f816979b61aa4ccd38e2912eff` |`job-df23nkw7ne5j` |35,558 |1,851 |
| **Eight terminal runs total** |Seven successes plus one failed setup | |**297,738** |**12,601** |

These are unique terminal-component costs, excluding original source
pretraining. The failed zero-action setup is counted ONCE; successful R2 is
its retry, not another20k collection. Conservative charged clocks are retained,
not replaced by shorter scheduler UI durations. No hold/refund/reallocation
occurs in this inventory update.

## Work that remains separate

The task0/40 long PixelSAC continuations were chosen AFTER the six-case onset
outcomes. The receipt's23:54:45 snapshot is outcome-selected development
diagnosis, not completion or confirmatory evidence. It intentionally does not
hash the changing checkpoints or claim final GPU cost. Their copied prefixes
are not new data; the parent35,558 actions /1,851 GPU-seconds are counted once.
No live status, final outcome or additional cost is inferred here.

Future RoboTwin wrapper guard `c166171b847ada9a20a9629a07d5a0100b517097`
remains source-prepared and staged for the execution owner's shared CPU gate.
No final allocated Linux gate receipt is supplied by this LIBERO handoff, so
qualification/admission/live deployment remain PENDING/NOT ADMITTED. The
existing `2fd2096` panel is not migrated, interrupted or replayed. Source
staging, static review, CPU qualification and live deployment remain distinct.
