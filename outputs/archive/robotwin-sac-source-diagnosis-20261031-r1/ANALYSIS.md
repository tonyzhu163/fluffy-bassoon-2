# SAC source95: weak learning is not specific to PPO

**Superseded by [r3](../robotwin-sac-source-diagnosis-20261031-r3/ANALYSIS.md), the source100 analysis.**

**PARTIAL: closed audited slice0055, source round95.** No source100 or final target-completion result is asserted. The named child endedexit0 with `verified_wall_budget_stop`, a complete round boundary,570completeK4trials and no pending trials. Its summary/file SHA and copied closed-child receipt are in `metadata-summary.json`. The active child, current checkpoint tensors, replay, frames and model parameters were not inspected or controlled.

Source JSON/JSONL records reconcile exactly to **2280episodes,954873actual actions,30844chunk decisions**. There are **257/2280successes(11.27%)** and2023failures, all recorded returns0or1. No conflicting inherited metadata identities were found. The training blocks score **78/600(13.0%),57/600(9.5%),76/600(12.67%),46/480(9.58%)** over rounds1–25/26–50/51–75/76–95. There is no useful upward trend through this boundary.

|Source goal|All95rounds|First25rounds|Rounds76–95|Actual actions|Decisions|
|---|---:|---:|---:|---:|---:|
|handover_mic|53/380|18/100|3/80|207987|6605|
|pick_diverse_bottles|13/380|5/100|4/80|148409|4827|
|place_a2b_left|51/380|16/100|10/80|140126|4567|
|place_object_stand|93/380|22/100|19/80|129129|4229|
|place_phone_stand|35/380|11/100|9/80|142644|4643|
|scan_object|12/380|6/100|1/80|186578|5973|

Source validation at the already-recorded iterations10/25/50 is **1/24,1/24,3/24**. Round100 validation is absent at this boundary and remains pending. The panel uses the previously reviewed fixed source seeds/native horizons; its small size and stochastic actions still limit generalization conclusions. A large count of optimizer updates does not substitute for demonstrated source success.

## Recipe and what an update means

This is the declared **feature-conditioned DSRL-SAC-style baseline**, not a reproduction of the authors' PixelSAC/DSRL-NA. It shares the frozen Motus pi0.5 motor,2062public features and full32×32=1024D bounded initial-noise interface with the current PPO campaign. Its actor/critic are128×3 MLPs with LayerNorm; twin-Q, AdamLR3e-4 for actor/critic/temperature, batch256, UTD20, replay100000, gamma.99, target tau.005, no gradient norm cap. Warmup is bounded uniform noise and begins SGD after500decisions. This differs from the published tied32D/PixelSAC and Gaussian-warmup recipe. Source SAC updates online after each new decision, including within an episode; PPO holds its actor fixed across the six-trial rollout group.

The closed receipt has **606880joint SAC updates**. Each joint update makes one actor, one twin-critic and one temperature optimizer step: **606880each,1820640combined optimizer calls**. Twin-Q is one combined critic optimizer step, not two. The count exactly matches **20×(30844−500)**. The source replay has only30844unique decisions at this boundary; replay optimization reuses them. These counters must not be described as606880new robot samples or compared directly to PPO100rollout updates/500AdamWsteps.

Only **1213diagnostic snapshots** are logged, at source decision multiples25, spanning joint update500to606500. Each reports the last SGD step in that decision's20step group, not an average of20steps or all606880updates. Per-block diagnostic attribution uses the exact cumulative source chunk/update clock. Replay minibatches mix goals, so these losses are not goal-specific. All recorded scalar metrics are finite; this does not prove every unlogged intermediate was benign or that the learned policy is useful.

## Available numeric diagnostics

|Logged quantity|First snapshot(update500)|Last snapshot(update606500)|Interpretation|
|---|---:|---:|---|
|Temperature alpha|0.859963|5.23599e-05|Autotuned entropy coefficient, not action std|
|Entropy estimate|699.25|-437.386|Joint normalized-coordinate differential entropy|
|Soft Bellman target mean|610.428|30.6399|Includes entropy bonus/bootstrap, not success probability|
|Critic loss|692478|0.00502565|Twin-Q summed MSE|
|Actor loss|-623.323|-30.9664|alpha*logprob minus minQ|

There is a large early soft-value transient: the maximum logged target mean is610.43 and maximum critic loss692478.5. Across rounds1–25 the median critic loss is0.1086 despite the mean8712.4, so a global mean alone is dominated by early outliers. The final20rounds' median critic loss is0.01697 and median soft target31.83. Later smaller Bellman error does not imply better robot success; actual success remains weak.

Temperature decreases from its configured initialization1 to values around5e−5 near the boundary; its logged minimum is1.90e−5. The default target entropy is **−1024**, since no override is given. Log densities sum1024dimensions with tanh Jacobian correction in normalized coordinates; external1.5scaling is excluded. Reported entropy is `−mean(logprob)`, **not divided by1024**. Its full logged range is−4063.92to700.26; the final20rounds' median is−968.19, while the last snapshot is−437.39. Negative differential entropy is valid. Alpha is not actor variance, and no action-std/saturation statistics were extracted, so “exploration collapsed” is not established.

The target is `reward + .99*(1−terminated)*(minQ−alpha*next_logprob)`. Time-limit truncations bootstrap, genuine success termination stops bootstrap. Soft-Q values therefore include entropy and imagined continuation beyond a timeout; they are not calibrated probabilities of success within the benchmark cutoff. The source sparse task reward remains0or1. The early value/temperature transients are evidence-supported optimization observations, not a demonstrated cause of failure or a new guard/recipe requirement.

## Comparison with the completed PPO arms

FULL source100 scores416/2400(17.33%); trainedRESET421/2400(17.54%), both flat across their quarters. SAC95's11.27% and flat blocks show that **weak source learning also occurs in this distinct reactive SAC learner**. This is not uniquely a recurrent-PPO source-state-transfer failure. These endpoints have different round counts and exploration/optimization objectives; the numbers do not establish a statistically reliable ranking or a shared mechanism.

PPO's lowLR/norm clipping cannot by itself explain SAC's weakness: SAC usesLR3e-4, no norm cap and606880joint updates. Shared motor/task/feature/noise interfaces are possible common factors; each algorithm could also have separate optimization issues. Neither possibility is isolated. No conclusion about original published DSRL, all frozen-VLA steering, in-context memory or target superiority follows from this partial source readout. Source100and final matched targets remain pending; no new run, resource request, checkpoint selection or launch gate is introduced.

Code identities and exact formulas: `vla_noise_sac.py:69–96,151–178,183–193,321–375`; the actual SAC executor’s `robotwin_noise_compare.py:631–659`. SHA-bound actual executor files are recorded, with scientific source`db0184089fa88a5f96c153f9b80029bb91b77def` and explicit later execution migrations retained. `extract_metadata.py` imports only standard-library modules and reads the original source JSONLs plus named closed slices0001–0055. Physical costs always use the canonical closed receipt; metadata deduplication never removes repeated physical execution charges.
