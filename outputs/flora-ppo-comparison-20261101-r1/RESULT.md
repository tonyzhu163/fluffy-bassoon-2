# Task44 baseline ceiling stops the proposed Flora PPO comparison

2026-11-01, baseline terminal at15:51:03 Shanghai. Flora fixture qualification later failed before scoring; PPO has not launched.

The matched comparison was intended to measure learning from simulator terminal
success versus Flora graded terminal feedback. Its predeclared admission rule
required a fresh native success rate between5% and70%, plus Flora AUROC at least0.7.
The published `RLinf/RLinf-Pi05-LIBERO-SFT` checkpoint at `45ccfcc4` instead
succeeded on **64/64** planned baseline episodes. Both PPO arms remain unadmitted.

The fixed bank covers40 nominal TRAIN placements0..39, with a second private
draw on the first24. All64 episodes ran the full400-step horizon plus15 settling
actions: **26,560** new actions,0 padding and0 SGD. The single separately planned
smoke added415, bringing common preparation to **26,975** actions. This is selected
development evidence with reused placements, not population or unseen-skill performance.
The current release differs from the historical C565 initializer used to select task44;
that older model's headroom cannot be transferred to this model.

With no simulator-negative episode, Flora AUROC is **undefined**, not passing or
zero. The already-running scorer may finish its archived smoke-video qualification,
but this does not establish label separation or justify PPO. There is no F-versus-S
learning curve, optimization result or convergence claim. More task44 training would
not resolve the observed baseline ceiling. At this terminal boundary, no task substitution, new baseline, score tuning
or search was admitted. The later human decision below is a separate prospective scope.

Baige baseline `job-xcadsqli2spo` succeeded with wrapper exit0. Its four-GPU clock
charged245 inclusive wall-seconds, **980 GPU-seconds** total. Closed comparison cost
is **1,544 GPU-seconds and287 CPU-worker-seconds** before the active Flora allocation
is settled. All earlier setup failures are retained. Loading3127, obsolete conversion3509,
and baseline13420 unused GPU-seconds are closed and are not transferred. Only the
existing Flora fixture ticket remains reserved, at most3600 GPU-seconds and0 actions.

Owner evidence is pinned to `c9f2e8c6`; runtime science remains `dd773048` and
resource wrappers are `2816e43f`. Root independently checked saved case metadata,
frame clocks, declared byte hashes and worker-clock totals. Root did not load models
or numerical arrays, decode videos, restore a checkpoint or replay remote journals.
See [parent review](BASELINE64-PARENT-REVIEW.json) and
[exact owner terminal packet](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/a5bc35893d2b156a1612784c1ac59bcac9ddcca6/outputs/flora-ppo-runtime-readiness-20261101-r1/baseline64-terminal-c9f2e8c6/BASELINE64-FINAL-RECEIPT.json).

## Subsequent human steering

After this result, the human explicitly requested a headroom search within LIBERO90,
with RoboTwin or RoboCasa as conditional alternatives. The separately charged
[prospective search](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/a5bc35893d2b156a1612784c1ac59bcac9ddcca6/outputs/flora-ppo-libero90-headroom-search-20261101-r1/PROTOCOL.json)
uses the same45cc checkpoint, fixed screening/confirmation rules and no PPO. It
does not change task44 results, refill its spent preparation budget, or select
a new task from Flora scores. No new task or benchmark has been selected yet.

## Flora fixture and revised direction

Flora job `job-8ccnyqvoj5fa` ended Failed at15:54:19 Shanghai. It loaded all11
weight shards (reported51.1GiB) and compiled before FlashInfer warmup rejected
the detected SM12/toolkit combination. Torch CUDA13 and allocation probing had
passed. The selected JIT nvcc path/version was not captured, so the precise
compiler selection remains unresolved; this is not a demonstrated weight or
out-of-memory failure. No label, physical action or optimizer update was produced.
Its446 GPU-seconds bring closed comparison cost to1990 GPU-seconds, with287
CPU-worker-seconds and26975 common actions. All allocations are terminal. The
same3600-second Flora ticket retains3154 seconds, with no automatic retry.
[Saved failure review](FLORA-Q1-TERMINAL-PARENT-REVIEW.json).

The human then prioritized restoring RoboTwin and its six VLRLA source tasks,
starting with `scan_object`, for a measured clean/random condition contrast.
The unlaunched LIBERO search is held with zero charge. Existing RoboTwin code
and archived results are present; native PPO/common-policy, video-clock and
finite runtime bindings remain to be prepared. The cached Motus policy saw both
clean and randomized demonstrations, so this is not an unseen-domain claim.
[Restoration scope](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/a5bc35893d2b156a1612784c1ac59bcac9ddcca6/outputs/robotwin-flora-six-task-restore-20261101-r1/AUTHORITY.json).
