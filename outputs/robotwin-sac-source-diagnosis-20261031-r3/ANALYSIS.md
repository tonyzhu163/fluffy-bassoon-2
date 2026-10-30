# Final SAC source100: weak learning is shared across all three arms

**Source training100 is complete; final evaluation is pending.** This immutable follow-up uses the audited closed slice0060,exit0,complete round boundary. It leaves the earlier95/98artifacts unchanged. A separate metadata-only read of the subsequent stage100manifest records `status:evaluating`; no actor/checkpoint tensor bytes were opened and no active process was inspected or controlled.

The closed source has **100rounds,600completeK4trials,2400episodes,1003790actual actions,32425decisions,638500joint SACupdates**. All episode/action/chunk totals reconcile, with no conflicting inherited metadata. Each joint SACupdate steps actor/twin-critic/temperature once: **638500each,1915500combined optimizer calls**. The exact count is20×(32425−500); it is replay optimization over32425source decisions, not638500new robot samples.

SAC succeeds **277/2400(11.54%)**. Its quarter totals are **78,57,76,66 /600** (13.0%,9.5%,12.67%,11.0%). The final five rounds add20/120successes versus the95snapshot, which does not change the weak overall/non-increasing source trend. Performance varies across goals; phone/object improve from first to last quarter while handover/scan decline.

|Goal|SAC /400|FULLPPO /400|TrainedRESET /400|SACfirst→last25rounds(/100each)|
|---|---:|---:|---:|---:|
|handover_mic|54|76|76|18→4|
|pick_diverse_bottles|14|40|35|5→5|
|place_a2b_left|54|70|66|16→13|
|place_object_stand|100|130|138|22→26|
|place_phone_stand|43|84|86|11→17|
|scan_object|12|16|20|6→1|

Completed FULLPPO scores416/2400(17.33%) and RESET421/2400(17.54%), also without an upward quarter trend. All three now have the same100goal-balanced rounds and2400episodes, with differing actual duration/decision counts. SAC's source result is descriptively lower, but this is one training seed, different exploration/objectives and an evolving source-state sample stream; it does not establish a statistically reliable algorithm ranking or a common failure mechanism.

## Identities and unfinished readout

The closed source-summary path/SHA and copied audited child receipt are in `metadata-summary.json`. Its declared source recovery SHA is retained from the closure receipt, without rehashing checkpoint tensors. The fixed stage100source actor snapshot is:

`/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-continuation-sac-20261030-r1/slices/0061/experiment/stages/iteration-100/sac-recovery.pt`

Declared checkpoint SHA: `59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90`.

The recorded stage has source actions1003790,source updates638500,pending trials0,source iteration100. `stage100-at-read.json` captures the manifest bytes/hash **at read time**; its status field may later transition from evaluating. The closed0060summary contains only prior complete evaluation stages10/25/50. **SourceVAL100,target100and wholecampaign exit remain unclaimed/pending**. Earlier sourceVAL totals are1/24,1/24,3/24. Completed source training must not be presented as completed final comparison.

## Diagnostics through the final source boundary

The1277logged snapshots span joint updates500–638500. They record the final SGD in each20stepgroup at every25thsource decision; statistics are not averages of every SGD step. Code/recipe remains the previously reviewed feature-conditioned1024D DSRL-SAC-style interface, not the authors'32D PixelSAC/DSRL-NA reproduction. No model or new code audit was performed for this refresh.

Final logged alpha is3.61517e-05,joint normalized-coordinate differential entropy-713.3589,soft Bellman target mean24.4805,critic loss1.5613and actor loss-25.3584. The full logged series is finite and includes the previously recorded early large critic/soft-value transient. Differential entropy can be negative; alpha is not action std. SoftQ includes entropy and timeout bootstrapping, so these values are not success probabilities or evidence of useful competence. Lower Bellman error during parts of training did not accompany a sustained robot-success gain.

This closes the specific source-learning question: weak source competence is **not unique to recurrentPPO**, nor solely a novel-target transfer issue. SAC does not share PPO's LR1e-6/norm1clipping, so those settings cannot by themselves explain all three outcomes. Shared feature/noise/motor/task interfaces and separate algorithm-specific optimization failures remain hypotheses. This evidence does not identify a causal remedy, establish failure of publishedDSRL, or support memory gains. The final matched source-validation/target results are still required; no checkpoint choice, new experiment, resource request or launch gate follows.

Only named closed JSON/JSONL metadata and the stage-manifest JSON were read. `extract_metadata.py` and the bound previous PPO summaries make this comparison reproducible without raw frames, replay buffers, policy construction, tests, inference or allocations.
