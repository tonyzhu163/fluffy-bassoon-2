# Independent audit of the two-task confirmation

The uncorrected IIWA demonstration prefix rescued one of six fresh drawer starts and lost two of six moka starts relative to standard tuned control. The earlier drawer panel had four rescues and zero losses. The new drawer result has the same direction and a smaller effect. The moka result weighs against a benefit across both tasks. I reproduced every paired effect from the original native step records. [Independent reduction](audit/gpu01.json) · [Original episodes](evidence/native01/episodes.tar.gz) · [Earlier eight-drawer audit](../frozen-transfer-demonstration-20261124-r1/READBACK.md)

## Outcomes on the registered panels

Numbers in each cell give the released row IDs that reached native success by the task horizon.

| Task | Standard tuned | Target raw then tuned | Direct IIWA then tuned | Corrected IIWA then tuned |
|---|---|---|---|---|
| Drawer, 300 controls | 3/6 · 21, 23, 24 | 0/6 | 4/6 · 21, 23, 24, 25 | 3/6 · 21, 22, 24 |
| Both moka pots, 520 controls | 2/6 · 21, 22 | 1/6 · 24 | 0/6 | 3/6 · 22, 23, 24 |

Each comparison below subtracts the comparator from the direct IIWA condition. Intervals are the 2.5th and 97.5th percentiles of all 46,656 ordered resamples of six paired rows within a task. They describe variation across the observed layouts. There is one noise seed per task-layout and two prespecified tasks.

| Task | Comparator | Rows rescued | Rows lost | Net success difference, percentage points (95% interval) |
|---|---|---|---|---|
| Drawer | Standard tuned | 25 | — | +16.7 (0 to +50.0) |
| Drawer | Target raw then tuned | 21, 23, 24, 25 | — | +66.7 (+33.3 to +100) |
| Drawer | Corrected IIWA | 23, 25 | 22 | +16.7 (−33.3 to +66.7) |
| Both moka pots | Standard tuned | — | 21, 22 | −33.3 (−66.7 to 0) |
| Both moka pots | Target raw then tuned | — | 24 | −16.7 (−50.0 to 0) |
| Both moka pots | Corrected IIWA | — | 22, 23, 24 | −50.0 (−83.3 to −16.7) |

Delaying calibration alone solved no drawers. With the same demonstration in both branches, bypassing calibration gained two drawer rows and lost one. On moka, correcting the demonstration produced three successes from the same six starts where direct replay produced zero. Corrected versus standard tuned on moka gained rows 23 and 24 and lost 21, a net +16.7 points (−33.3 to +66.7). As a secondary calculation, the pooled direct-versus-tuned effect was −8.3 points (−33.3 to +16.7), holding six draws per task. All six pairwise contrasts per task and the stratified pooled contrasts are recorded in the reduction. [Rows, pairs and intervals](audit/gpu01.json)

## Donors, tasks and matched starts

- The original raw IIWA drawer episodes on rows 34–37 succeeded in 171, 184, 248 and 251 controls. The shortest successful source was row 34. Its original cue-recording bytes matched the archived donor copy. The registered moka source was the raw IIWA row 31 recording from the progress study. It succeeded at control 361. Both source reward series were zero until a terminal reward of one. For each donor I matched all seven issued float32 channels against the clipped raw proposal up to termination. [Donor and support evidence](audit/references.json)
- The source gripper first became positive at zero-based indices 106 and 107 for drawer and moka. Rounding down to a completed five-control block gives 105 for each. Controls 1–105 end before either closing command. The sign crossing defines a command boundary. Contact or a physically completed approach was not measured by that rule. Donor identity and cutoff stayed fixed across all target rows and conditions. [Registration](registration.json) · [Original command digests](audit/references.json)
- I derived fingerprints for rows 20–25 of the released BASE state bank for each task. The reset loads the native Panda scene, transfers the scene and fingers to mounted Kinova3, aligns the grip site with joint-limited IK, and settles before policy control. All four conditions shared the same released row, settled physical-state and scene hashes, reset anchor, cameras, proprioception, opening model input and proposal. All 48 resets were accepted on their first attempt. The official drawer predicate checks the bowl in the top drawer region. The kitchen predicate requires both pots on the stove's cook region and the stove turned on. [Release, BDDL and bank readback](audit/references.json) · [Paired starts](audit/gpu01.json)
- The frozen seed-41 tuned Perceiver prediction came from a separate nominal Kinova3 layout-11 passive support of 15 controls per task. Drawer used +4.2834° yaw and 0.92060 scale. Moka used +0.9162° and 0.95558. Both retained grip sign +1. The corrected condition applies this ordinary action inverse to seven-channel commands already issued by IIWA. Neither the source tape nor its task-specific demonstration trained or conditioned a new Perceiver in this experiment. [Support identities and coefficients](audit/references.json)

## What reached the model and simulator

The pinned runner loads OpenPI `pi05_libero` and passes current target external and wrist views, end-effector/gripper state, the released prompt and explicit noise to inference. It queries even while a donor prefix controls the robot. I independently hashed all 16 checkpoint files (12,439,085,481 bytes), regenerated each task-layout noise bank, and matched every global replan slice. I traced the selector to mode and elapsed controls. Simulator reset diagnostics, goal predicates, target rewards, other branches and future target frames have no path into its action choice. The physical wrapper retains native diagnostics for adjudication. Saved camera evidence consists of input digests, so camera verification uses the pinned preprocessing path and digest continuity. [Pinned runner](confirmation.py) · [Independent code and checkpoint hashes](audit/gpu01.json)

In the target-generated switch, all 1,260 prefix commands matched the clipped proposal generated from current raw target input. Raw and tuned input states differed at all 240 nonopening prefix queries across the twelve starts. At handback, all 36 switching or demonstration episodes reached global query 21, immediately before control 106. Each used the latest native frame, a tuned state distinguishable from raw, and the originally scheduled noise. All five native commands at each of those handbacks differed from the unconverted current proposal. Across all four modes, all 13,209 post-cut controls matched the reconstructed tuned inverse byte for byte and differed from their clipped unconverted motor command. [Per-mode and per-stage checks](audit/gpu01.json)

Every direct donor command matched the archived bytes. Applying the inverse to the same donor changed all three translation coordinates on all 105 prefix steps for each task. All rotation coordinates and grip values were equal. Clipping erased zero coordinate changes. The largest absolute command difference was 0.10830 for drawer and 0.02600 for moka. Both interventions reached the native simulator, including on rows where their outcomes tied. [Actual command differences](audit/gpu01.json)

## Terminal and resource checks

The independently read Baige originals cover 48 distinct episodes, 18,249 native steps and 3,656 frozen-policy queries from job `job-zyhf39n6wf5f`, run at `22cc4bd4caeb9a07bb2d69c745a1bf6aff522061` on one RTX 4090. The job exited zero in 1,036 seconds. I reconstructed each complete five-command proposal before slicing the portion actually executed. Fourteen terminal chunks ended after one to four controls. Every native success earned one on its last control, all preceding rewards were zero, and the other 32 episodes truncated exactly at their official task horizons. Every policy query followed the preceding chunk's actual observation. Each transported episode and launch receipt matched the SHA-256 read directly from shared storage. [Native records](evidence/native01/episodes.tar.gz) · [Transport check](audit/transport.json)

My audit used CPU cores 46/47 with GPUs hidden and launched zero simulations. The separate study-worker qualification had eight fixture records and six fixture handbacks. The independent reader rejected ten deliberately corrupted record copies for each task, including erased switch input or command, an incorrect source command, restarted noise, a fabricated reward and a step after termination. The terminal reduction used 9.7 CPU seconds. [Qualification audit](audit/cpu01.json) · [Read-only reader](audit/read.sh) · [Terminal audit](audit/gpu01.json)
