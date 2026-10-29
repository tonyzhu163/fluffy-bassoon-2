# Two-scene online DSRL-NA control audit

The completed online control scored 4/8 successful episodes on the first two
paired pilot scenes. Frozen DSRL-NA scored 7/8 on those same scenes; RL² normal,
its memory-reset lesion and the trained reset controller each scored 8/8.
Online adaptation did not improve this small paired evaluation. Two scenes
and one training seed do not establish a general method ranking or a broad
negative result for online adaptation.

The comparison unit is the initial scene: two paired scenes, seeds 2000000 and
2000001, with four retries each. The eight episodes are repeated measurements
within those scenes, not eight independent trials. All counts are descriptive.

## Matched scene outcomes

The bit strings list attempts 1–4, where 1 means success. Attempt counts are
out of two scenes; totals are out of eight recorded episodes.

| Configuration | Scene 2000000 | Scene 2000001 | Successes by attempt /2 | Any-step total /8 | Release-logger total /8 |
|---|---|---|---|---:|---:|
| Online DSRL-NA | 1100 | 1010 | 2, 1, 1, 0 | 4 | 4 |
| Native frozen motor | 1101 | 1010 | 2, 1, 1, 1 | 5 | 5 |
| DSRL-NA frozen | 1111 | 1011 | 2, 1, 2, 2 | 7 | 7 |
| Retry DSRL-NA frozen | 0001 | 0000 | 0, 0, 0, 1 | 1 | 1 |
| RL² normal | 1111 | 1111 | 2, 2, 2, 2 | 8 | 8 |
| RL² memory reset | 1111 | 1111 | 2, 2, 2, 2 | 8 | 8 |
| RL² reward blind | 1111 | 1011 | 2, 1, 2, 2 | 7 | 7 |
| Trained reset normal | 1111 | 1111 | 2, 2, 2, 2 | 8 | 8 |
| Trained reset memory reset | 1111 | 1111 | 2, 2, 2, 2 | 8 | 8 |
| Trained reset reward blind | 1111 | 1011 | 2, 1, 2, 2 | 7 | 7 |

The any-primitive-step and release-logger metrics agree at every scene and
attempt in this two-scene subset. Online minus frozen DSRL-NA is -2 successful
attempts on scene 2000000 and -1 on scene 2000001. Online minus RL² normal or
trained reset normal is -2 on each scene. Relative to native it is -1 and 0;
relative to retry DSRL-NA it is +1 and +2. The online attempt counts decrease
from 2/2 to 0/2; this run provides no descriptive retry-improvement curve.

These scenes do not differentiate RL² normal from either reset control.
The pilot's sole normal-versus-reset difference occurred on scene 2000004,
which is outside this optional control. Comparing online 4/8 directly with
the full pilot's 21/32 would change the scene set; the table above avoids that.

## Compute and trial isolation

Every row uses 2,400 deployment primitive steps, or 600 four-action chunks.
Frozen evaluations perform zero deployment optimizer updates and 600 motor
samples. The online control performs one complete released 20-gradient plus
10-distillation training call after every observed single-lane chunk:

| Deployment quantity | Online DSRL-NA | Each frozen comparator |
|---|---:|---:|
| Primitive simulator steps | 2,400 | 2,400 |
| Training calls | 600 | 0 |
| Actor optimizer invocations | 12,000 | 0 |
| Action critic optimizer invocations | 12,000 | 0 |
| Entropy optimizer invocations | 12,000 | 0 |
| Noise critic optimizer invocations | 6,000 | 0 |
| Total optimizer invocations | 42,000 | 0 |
| Motor batch calls | 18,600 | 600 |
| Motor samples, including target/distillation queries | 4,608,600 | 600 |

During original training the same 20+10 update call follows a four-lane vector
step. Applying it after each single-lane chunk online gives four times the
updates per new transition. Thus deployment interactions match, while
optimization, update-to-data ratio and decoder-query compute do not.
Adaptation already occurs within the first attempt. A common initial RNG seed
does not couple subsequent stochastic choices because online optimization
consumes random draws.

Each scene starts from the same 28,800-step original DSRL-NA recovery. Both
scenes record identical starting actor, action critic, target critic, noise
critic, combined optimizer and entropy hashes, plus 7,200 starting replay
transitions. Each then performs 300 training calls and 21,000 optimizer
invocations; final learner/optimizer/entropy hashes change independently.
No held-out adaptation is carried between scenes in the recorded evidence.

All 131 metadata/hash/count checks passed. The seven directly mirrored online
files and the paired pilot/native evidence match independently obtained remote
hashes. The current post-run hashes of `recovery.zip`, `replay.pkl`, `runtime.pt`
and the original recovery manifest equal both the online recorded reference
hashes and the pilot manifest. The pretrained motor checkpoint and
normalization file also retain their expected hashes. Online records
`reference_unchanged=true` and `motor_unchanged=true`.

Online source `2a02e6c7b165e37199df81ba46308d82caffdde2` differs from the
training source `103e0f5a1a58d2b7f09756ea74e81605d72cbcf6`, while all eight
shared source files hashed in `raw/source-manifest.json` are identical,
including the environment, driver, comparison helper, GTrXL, PPO, GPU helper
and probe. Runtime/input identities and the paired initial-scene hashes agree.
The allocated GPU probe passed on Baige `job-060f4odvylld` using one RTX 4090.
The completed allocation recorded 523.569 seconds, or 8.726 GPU-minutes; the
model driver recorded 436.977 seconds and 0.793 GiB peak allocated CUDA memory.

This audit used stdlib artifact reads, counts and hashes only. It did not load
model, optimizer or replay tensors, run tests or simulation, or copy checkpoints
to macOS. Reference checkpoint hashing ran on Baige. Fresh replay content was
not independently inspected; its isolation evidence is the fresh-load protocol,
identical recorded learner/optimizer hashes and starting transition counts.

`RESULTS.json` contains both metrics, all attempt matrices, paired differences,
compute budgets and checks. `derive_results.py` reproduces it from `raw/`.
`raw/remote-manifest.json` records raw evidence and post-run reference/input
hashes; `raw/source-manifest.json` binds the pinned source identities.
