# K4 attempt rates do not reveal a hidden memory gain

Descriptive source analysis,2026-10-31. All three named immutable source100boundaries reconcile to **2400source episodes and600completeK4groups each**. Success/action/chunk totals exactly match the prior FULL/RESET/SAC source summaries. Every round contains each of the six source goals once; every(goal,round)group has exactly the saved ordinals1,2,3,4 and four distinct scheduled scene seeds. No missing groups or conflicting inherited receipts were found. Costs are unchanged; metadata deduplication never removes physical interactions.

|Method|Round subset|Attempt1 successes|Attempt2|Attempt3|Attempt4|Episodes per attempt|
|---|---|---:|---:|---:|---:|---:|
|ppo|All100|103|106|97|110|600|
|ppo|First25|31|32|27|26|150|
|ppo|Last25|24|25|28|26|150|
|ppo_reset|All100|98|110|106|107|600|
|ppo_reset|First25|21|37|33|21|150|
|ppo_reset|Last25|26|22|27|26|150|
|sac|All100|72|70|64|71|600|
|sac|First25|17|21|14|26|150|
|sac|Last25|21|13|16|16|150|

Across all100rounds, FULL rates are **17.17%,17.67%,16.17%,18.33%**. RESET rates are16.33%,18.33%,17.67%,17.83%; SAC12.0%,11.67%,10.67%,11.83%. FULL's attempt4minus1is+1.17percentage points; RESET's+1.50points is at least as large; SAC's−0.17points. These small non-monotonic differences do not establish within-trial adaptation or a FULL memory advantage. FULL's first25rounds actually decline from attempt1to4(31→26/150); its last25have24→26/150 with a peak at attempt3. There is no strong monotonic hidden improvement behind the flat outer learning curves.

Per-goal totals below are successes on attempts1/2/3/4, each denominator100. Heterogeneity is visible, but no consistent FULL-specific later-attempt gain emerges.

|Source goal|FULL|TrainedRESET|SAC|
|---|---:|---:|---:|
|handover_mic|21/15/17/23|17/21/14/24|12/15/9/18|
|pick_diverse_bottles|9/12/9/10|7/10/13/5|3/3/4/4|
|place_a2b_left|17/20/14/19|14/21/16/15|13/21/12/8|
|place_object_stand|35/31/33/31|37/29/38/34|28/22/26/24|
|place_phone_stand|19/21/20/24|19/23/19/25|13/8/11/11|
|scan_object|2/7/4/3|4/6/6/4|3/1/2/6|

## What an ordinal/group means

`ordinal-code-evidence.json` binds exact actual executor source bytes and narrow relevant spans. FULL's parallel collector initializes each lane's ordinal at0, logs `episode+1`, increments after completed episodes, and chooses the corresponding next seed. Serial/recovered collection enumerates with the original `start_episode`, preserving ordinal on continuation. The balanced task scheduler samples a shuffled six-task block and four episode seeds for each selected task. PPO metadata also preserves a constant `trial_in_iteration` across the four episode records. SAC logs `source_iteration`,goal and1-based episode; its(goal,round) grouping is unambiguous because the six-goal complete schedule and exactly four rows/group reconcile.

The four attempts share a task goal, **not a verified identical physical scene**. Their four scheduled seeds differ; different seeds do not themselves prove distinct physical states. Differences across ordinal can reflect scene variation, stochastic draws and the explicit episode-index clock. The source group definition is therefore not a same-case support/query protocol.

PPO weights are fixed during the six complete K4trials of an update. FULL retains interaction history across the four episodes. TrainedRESET clears cache and previous-noise/reward feedback at the episode boundary but retains within-episode recurrence and an episode-index clock. Thus ordinal effects can exist without cross-episode outcome memory. The reactive SAC learner changes parameters online within episodes and between attempts; any later-attempt improvement there would combine SGD/exploration/state variation, not in-context recurrence. These controls do not permit a causal memory comparison from source marginal rates alone.

The records contain FULL416,RESET421,SAC277successes, matching previous source100analyses. FFFFpatterns account for318/600FULL groups,305/600RESET and397/600SAC. Some groups have failures followed by successes, but because their seeds differ, these patterns must not be labeled same-case recoveries. No target episodes enter this analysis, no support/query budget is inferred, and no new benchmark result or run is created.

`inputs.json` hash-binds the three prior summaries and every named closed episode JSONL. `extract_metadata.py` uses only standard-library receipt processing and refuses missing/malformed groups, changed source bytes or cost/success mismatches. `attempts.json` contains all counts, per-goal first/last25strata and outcome patterns. No actors, tensors, replay, frames, models, simulator, tests, allocations or active processes were used. No new pipeline gate or experimental recommendation follows.
