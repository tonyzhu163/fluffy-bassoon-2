# Native CTM warm-start pilot completed

Baige `job-l3coajmm9noh` completed the approved native recovery pipeline with
worker and launcher exit zero at **2026-10-28 18:46:56 Asia/Shanghai**. Source
remained `79da19500f4e898d078539b4c562c150e610c16a`. It saved a distinct native
CTM candidate, evaluated it with frozen weights, and loaded it into one actual
PPO update. This qualifies execution and checkpoint transfer; useful reward-driven
adaptation and a benefit over direct PPO remain unestablished.

## Label-policy competence and pretraining

The frozen motor scored **6/16 successes (37.5%)** versus **15/16 (93.75%)** for
the privileged inverse-rotation label policy on four paired training-distribution
trials per policy. This passes the predeclared small competence gate. It is an
oracle result, not the learned CTM's score or evidence of statistical superiority.

Two complete history batches contain four trials / sixteen attempts / 1,280
simulator steps, with 256 additional data-label motor calls. Their SHA256 hashes
match the saved manifest. Batch one trains the CTM at rotations `[45, 0]`;
batch two validates at `[0, 0]`. Neither includes -45 degrees. The tiny realized
coverage is an execution screen, not adequate adaptation pretraining.

All **100 supervised optimizer updates** completed in 208.76 seconds. The
complete-batch validation flow loss fell from **.386777 to .333308**. This loss
alone does not validate behavioral adaptation. The saved data uses one fixed
flow-noise/time sample per history position.

## Frozen live behavior before PPO

The evaluation at **iteration zero**, before any PPO update, uses the separate
held-out rotation/overlay distribution declared in the run configuration. There
are only **two trials, four attempts each** per condition, with paired plans.

| Condition | Successes | Mean | Attempt means |
| --- | ---: | ---: | --- |
| Normal CTM | 3/8 | 37.5% | 0%, 50%, 50%, 50% |
| Memory wiped between attempts | 2/8 | 25.0% | 0%, 100%, 0%, 0% |
| Reward feedback removed | 3/8 | 37.5% | 0%, 50%, 50%, 50% |

Normal exceeds memory-wiped by one successful attempt and ties reward-blind in
both the aggregate mean and attempt curve. Individual paired outcomes of normal
and reward-blind differ; equal aggregates do not mean identical trajectories.
This is insufficient evidence of useful reward-driven ICRL. Do not promote the
candidate or equate its 37.5% held-out score with the earlier 37.5% frozen-motor
score: those evaluations use different task distributions and plans. There is
no matched zero-initialized CTM or direct-PPO comparison in this pilot.

## Transition to PPO

The recorded initializer SHA matches the exported native warm start. The first
rollout's stored likelihoods and critic values replayed with maximum absolute
error zero. One optimizer step ran with finite nonzero gradients on 111
parameter tensors; the KL guard then stopped the remaining epochs (`target_kl=.05`).
The precise KL value that triggered stopping is not retained by this source.
The accepted epoch's logged KL of zero is its pre-update value, not the final KL.

The first PPO collection scored 7/8 on training plans. That rollout precedes
the optimizer step and repeats training-distribution plans; it is not a post-PPO
performance result. The saved summary's `last_evaluation` is still the frozen
iteration-zero evaluation above. Full PPO recovery state was saved. Frozen
motor fingerprints were identical before and after both live stages.

## Artifacts and audit

Root: `/mowenjie/fluffy-bassoon-2/outputs/ctm-dpt-warmstart-20261028-r2`.

| Artifact under root | SHA256 |
| --- | --- |
| `native-warmstart/warmstart.pt` | `fb219453dd12e84d8e2f534ac17515543cd334c5bedf1858efd8a2fe584542d2` |
| `native-warmstart/latest.pt` | `c4d60ef8fd2c6f9700821e7cce0e3b6323943cf004316d7cf0539d0bfcc8236c` |
| `native-ppo-smoke/latest.pt` | `9e728afe29f161544cc258b461f759d96af6b115714a0afa1d7ea077df8f921d` |
| `native-collection/latest.pt` | `d00a09f9e29c262e6d783d6983e9d9b3a7c118d7ebb52303a3c110e9f73b67c2` |
| `native-collection/histories/manifest.json` | `3a0d41829d55403b6f0f058afb55349fd6fabe831fca4e0633dc3204299d2961` |

[CHECKPOINT.json](CHECKPOINT.json) records the native checkpoint family,
compatibility and behavioral status. Compact candidates remain separately
indexed in [r1](../ctm-dpt-warmstart-20261028-r1/CHECKPOINTS.json).

The completion audit checked seven remote artifact digests (including both
history shards), initializer identity, all five raw success arrays and attempt
curves, motor immutability, exact PPO replay and the KL stop. Raw receipts are
retained remotely and mirrored locally under `evidence/completed/`; `audit.py`
reconstructs the receipt without executing models or environments. Final
summaries account for **6,400 simulator steps** across collection, label-policy
evaluation, frozen candidate evaluation and the PPO rollout. Collection's
iteration-two metric snapshots counters before its evaluation; use the final
summary for the full 2,560 label-gate evaluation steps.

The approved bounded pipeline is complete. The completion monitor is retired;
no follow-on training is launched by this audit.
