# Native history experiment execution

Authorized November 17: run the small native RoboCasa experiment to determine
whether history helps. This is a separate phase with an **8 GPU-hour working
budget**, including qualification, collection and small reader fits. It does
not reopen the earlier 230 GPU-hour campaign. Account for the existing gate
from its process start at 11:09:33 Shanghai.

The allocated Baige development container has one complete RTX 4090 with a
100% compute allocation. The gate's actual CUDA probe passed; no other GPU
process was present before it began. The project mount has about 93 GiB free,
and a write/read/delete probe passed. No quota utility is available. Collection
keeps compact action tapes and public-state/reward records, rather than full
camera streams. Shared environments and checkpoints stay in place.

## Immutable qualification already running

- Source: `38eaf99bcc17d3930dab99aea4170fc0842e9c67`.
- Checkout: `/mowenjie/fluffy-bassoon-2/sources/feedback-advantage-38eaf99b`.
- Output: `/mowenjie/fluffy-bassoon-2/outputs/robocasa-feedback-advantage-20261117-r1/gate01`.
- Development hostname: `aibox-rffe453e3884-5f96c78b88-6ssrn`.
- Controller PID 319421; collector PID 319472, verified against exact commands.
- First scene reset fingerprints match exactly; first episode is underway.

Do not mutate this checkout or reuse the output identity. Inspect the native
qualification before launching the sixteen-scene panel. The two gate scenes
are development cases; if the panel proceeds, they remain in its training
portion. Failure of this calibration intervention is a negative qualification
result, not a negative result for every history learner.

## Changes before any later learner fit

Review found that NumPy advanced-index increments undercount repeated samples
inside a minibatch. Use `np.add.at` for the actual exposure inventory. All
methods now receive the same sampler seed within a fit seed, matching the
declared comparison. These changes do not affect the active collector.

The implementation evaluates three possible query corrections (-1, 0, +1),
so the probed suite contains 768 constructed training examples and the drift
suite 384, assembled from a much smaller set of measured native branches.
They are not hundreds of independent native trials. Final update 1024 is the
fixed endpoint; intermediate held-out curves do not select the winner.

The native branch table exhaustively measures each candidate correction at
the paired decision state. Scoring a frozen reader against it is a finite
action-selection evaluation using native outcomes, not a new closed-loop
VLA rollout or evidence that the previous large KV adapter now learns.
The opposite-calibration donor is a controlled contradictory-history
intervention, not a random unrelated-history baseline. The mask is an
evaluation-only input intervention and may be out of the training distribution.
Small scene counts and shared support/query scenes limit statistical precision;
pairing dependencies must be retained in the final readout.

Before fitting, score Stage 1 separately. Bootstrap blocks join every scene
that appears as a query, support or donor in a shared trial. The primary
eight-scene cross-scene panel has only two independent four-scene blocks;
its intervals are descriptive and coarse. The all-to-all panel has one
connected block and reports no confidence interval. This avoids treating
recombined native branch outcomes as independent episodes. Save complete
actor/optimizer/sampler/RNG snapshots at each declared training milestone.
