# PPO100 full/reset: weak source learning, no established memory gain

JSON/JSONL-only follow-up,2026-10-31. The RESET diagnosis is unchanged. The new FULL terminal receipt is `robotwin-trained-continuation-20261031-r1/ppo/slice-043/experiment/summary.json`; its SHA and all contributing metadata/code SHAs are in `metadata-summary.json`. No actors, tensors, raw frames or replay buffers were loaded; no tests, models, simulator, allocation, source edits or new recipe were run.

FULL reconciles **100updates,600completeK4trials,2400episodes,500AdamWsteps,964875source actions and31211valid chunk decisions**, with no missing updates or conflicting duplicate metadata. RESET has the same iterations/episodes/optimizer count,963759actions and31172valid decisions. FULL adds1116actions and39tokens—exposure is close, but these are separate trained policies and histories.

Source training success is **FULL416/2400(17.33%),RESET421/2400(17.54%)**. In successive25-iteration blocks, FULL scores **116,104,93,103 /600**; RESET **112,106,102,101 /600**. Neither shows a useful upward learning trend. Novel target tasks alone cannot explain this: source training is already weak. Heldout-source-state transfer is not the sole observed issue.

|Source goal|FULL / RESET training successes(/400each)|FULL first→last25iterations(/100each)|RESET first→last25iterations(/100each)|
|---|---:|---:|---:|
|handover_mic|76 / 76|20→18|20→17|
|pick_diverse_bottles|40 / 35|11→10|12→10|
|place_a2b_left|70 / 66|21→15|22→12|
|place_object_stand|130 / 138|37→36|31→34|
|place_phone_stand|84 / 86|23→18|24→19|
|scan_object|16 / 20|4→6|3→9|

Fixed source validation at iterations10/25/50/100 is **FULL1,2,4,2 /24** versus **RESET4,4,2,3 /24**. Final FULL per-goal scores are0/4,1/4,0/4,1/4,0/4,0/4 in the above order. Four fixed seeds per source goal and stochastic actions make this a coarse panel, not a broad source-competence or regression estimate. Source goals are seen; scene seeds are independent from training. Native task horizons and procedure remain the previously reviewed ones. All1984FULL failures reach their declared horizon, and all FULL training returns are0or1.

FULL numerical metrics resemble RESET: mean logged approximate joint KL **0.000226** (maximum0.001530), mean clip fraction0.000141, median clip fraction0, mean pre-clipping gradient norm66.25 (range55.50–79.42, all above norm1). Mean total/actor/critic losses are0.05123/−0.00864/0.05988; all logged metrics are finite. FULL batches average312.11valid decisions, range267–348. These establish actual finite optimization with conservative movement; they do not establish useful robot learning. As in the reset artifact, log-probs sum all1024noise dimensions, then `(exp(log_ratio)-1)-log_ratio` averages valid tokens and epochs. There is no1024dimension divisor and no separately measured final-policy analytic KL in these logs.

## Frozen target endpoint

All rows below use the predeclared8scene-seed panel per task at iteration100; motor/controller parameters remain frozen. These are eight sequential scored episodes, not a new sample-size estimate from eight training seeds.

|Condition|adjust_bottle|move_stapler_pad|Total|
|---|---:|---:|---:|
|FULL normal|7/8|1/8|8/16|
|Same FULL weights, cache reset per episode|6/8|1/8|7/16|
|Same FULL weights, reward input zeroed|8/8|1/8|9/16|
|Separately trained episode-reset policy|8/8|0/8|8/16|

There is **no demonstrated systematic memory or reward-use advantage**. Normal FULL equals trainedRESET overall; its one bottle success over the same-weight cache reset is offset by rewardblind performing better. Counts this small cannot support a robust ranking. The two stapler successes in normal/cache-reset occur on different scene-seed episodes; equal aggregate counts are not the same successful trajectory. Stochastic action draws and prior episode lengths can also diverge across conditions, so matching scene-seed lists alone is not a fully paired action-RNG causal experiment.

The cache-reset intervention clears the recurrent cache, but FULL's direct previous-noise/reward feedback remains at a boundary because its `reset_each_episode` flag stays false. The trainedRESET clears both cache and previous feedback while retaining within-episode recurrence and an episode-index clock. These are distinct controls; neither should be described simply as a memoryless Transformer. Rewardblind zeros the reward input and leaves other history available. No target optimizer update occurs in these frozen PPO evaluations.

Why this is unexpected: nearly1Msource actions and100updates sound substantial, but both source policies remain around17%training success and receive only~312valid decisions/update. The already-recorded VLRLA8192nominal slots/update is a capacity reference for a different learned component, not a matched recipe or a causal explanation. Small batches, sparse successes, shared norm clipping/LR and a bounded full1024D noise distribution remain possible explanations, not identified causes. This result is insufficient to reject in-context steering in general or to justify more data as its remedy.

The next interpretation is deliberately held for the matched native/FULL/RESET/sourceSAC endpoint comparison. SourceSAC remains pending at this snapshot. No checkpoint selection, experiment, resource request or pipeline gate follows from this analysis.

`extract_metadata.py` reads only the named original and continuation JSON/JSONL streams, deduplicating inherited identities and reporting missing/conflicting updates explicitly. `comparison.json` binds both saved summaries. Physical action counts always come from the canonical cumulative source receipt, never from removing repeated physical executions.
