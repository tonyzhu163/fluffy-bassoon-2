# A source-derived delay in vertical motion did not transfer after gripper closure

**Stop this adapter at the CPU diagnostic.** On two unseen simulated arms, the source-derived timing correction increased vertical motion error relative to static gain by 0.018 mm per control. The source-timed and two-control-shifted branches differed by only 0.0007 mm per control. Both timing and static gain completed 2 of 12 fixed-command replays. The shifted control completed 4. No GPU job was submitted. [Pre-registered question and literature](PLAN.md) · [Paired measurements](evidence/pairs.csv)

## Held-out native execution

Panda and Sawyer supplied the source fit on four development tasks and layouts 1 and 4. The law was frozen before evaluation on IIWA and UR5e, three different official test tasks, and layouts 14 and 15. For each held-out start, Panda and both target bodies received the same archived frozen π₀.₅ commands. The adapter operated on vertical translation for ten controls beginning at the first replan after commanded closure. The shifted control permuted the same corrections within each available five-action prefix. All twelve target windows were complete. [Source data and fit](evidence/fit/model.json) · [Native branches and trajectories](evidence/traces.tar.gz)

| Held-out arm and endpoint | Identity | Static gain | Source timing | Shifted timing |
| --- | ---: | ---: | ---: | ---: |
| IIWA vertical increment error, mm/control | 0.447 | 0.472 | 0.475 | 0.475 |
| UR5e vertical increment error, mm/control | 0.516 | 0.474 | 0.507 | 0.509 |
| **Both arms, mm/control** | **0.481** | **0.473** | **0.491** | **0.492** |
| Both arms, vertical path error from window start, mm | 2.345 | 2.269 | 2.351 | 2.347 |
| IIWA native completions within tape | 0/6 | 2/6 | 1/6 | 2/6 |
| UR5e native completions within tape | 1/6 | 0/6 | 1/6 | 2/6 |

Primary motion scores are per-window root-mean-square differences in realized vertical increments from the identically commanded Panda reference, averaged across starts. Positive differences favor the comparator. The timing branch had 0.010 mm/control more error than identity (paired 95% task-bootstrap interval −0.012 to +0.036), 0.018 more than static gain (−0.002 to +0.034), and 0.0007 less than shifted timing (−0.0114 to +0.0083). Timing minus static was −0.011 for moka pots, +0.030 for drawer and bowl, and +0.035 for cream cheese and bowl. The intervals enumerate resampling the three task identities, retaining both layouts and both arms together. With three tasks they provide limited precision about other tasks and condition on the source fit. [All contrasts, task effects and intervals](evidence/scores.json)

**The interventions reached actual contacts.** Force-bearing finger contact occurred in both the raw target and Panda during all twelve windows. On ten starts both arms touched the same task object. On drawer layout 15 Panda touched the bowl while IIWA contacted the stove and UR5e the table. In a descriptive check restricted to the ten same-object starts, timing, static and shifted movement errors were 0.4481, 0.4316 and 0.4481 mm/control. On all twelve starts every treatment changed all ten eligible Z commands, with zero clipping. The aligned and shifted differential components had identical per-prefix values up to permutation and the same mean window RMS of 0.0158 normalized command units. [Contact partners, intervention magnitudes and per-start results](evidence/pairs.csv)

**Native rewards were sensitive to the command change, but did not favor the source timing.** Timing gained IIWA drawer layout 14 over identity and preserved the one raw completion. Static gain also completed that drawer, gained IIWA moka layout 15, and lost UR5e cream cheese layout 15. Against timing, the shifted branch gained the IIWA moka start and UR5e cream cheese layout 14, with zero losses. Every branch had the same tape-length budget. All 39 target trials without a completion exhausted their donor tape before the released horizon. These outcomes measure completion on a common fixed continuation. Fresh target-policy replanning could change both the commands and the outcome. [Native rewards for every control](evidence/traces.tar.gz)

## What the fitted delay measured

Each of the eight source tapes supplied ten control-level pairs from fresh native Panda and Sawyer executions. The static fit gave gain 1.0272. The dynamic fit gave gain 1.0496 and first-difference coefficient −0.1561, equivalent to mixing in 0.149 of the previous control after normalization. At 20 Hz that corresponds to 7.4 ms of effective delay in this first-order model. Fitted error fell from 0.855 to 0.841 mm/control. Leaving out one of four source task identities reversed the inferred sign when the omitted task was libero_spatial/1. The other three omissions gave effective delays of 0.136 to 0.313 controls. Task-specific loading and trajectories can influence this coefficient. [Source sensitivity](evidence/scores.json)

On UR5e, timing improved the primary motion score over identity but lost to static gain. On IIWA it lost to identity. The near tie with the shifted correction and the absence of a native completion advantage over static leave no basis to advance this fixed source-derived delay. The comparison leaves open robot-specific latency measured independently on the target, other motion axes, and longer contact-dependent hysteresis. The tested input was a feedforward function of the available proposal and preceding command. Rewards, contact labels, Panda references and future replans were used only for stopping or analysis as appropriate. [Fitting and native execution](diagnostic.py) · [Scoring](score.py)

## Replay fidelity and reproduction

Every four-way within-arm comparison matched on full reset hash, state at the intervention boundary, public prefix positions and original prefix actions. Released scene fingerprints matched Panda across arms. All six Panda CPU replays completed the archived task. Their full reset hashes differed from the historical GPU resets. Five reproduced the full native reward sequence. Drawer layout 14 completed at control 173 versus 175. The largest window-specific CPU versus historical Panda vertical-increment discrepancy was 0.0125 mm/control on that start. Across all six full replays, maximum single-coordinate discrepancy was 0.407 mm. Rescoring against historical GPU Panda positions gave 0.4820, 0.4739, 0.4919 and 0.4926 mm/control for identity, static, timing and shifted. The ordering was unchanged. [Per-donor lineage, errors and archived-reference sensitivity](evidence/scores.json)

Baige Linux used cores 42 and 43, Mesa llvmpipe, the existing uv simulator environment, MuJoCo 3.2.3, robosuite 1.4.0 and zero GPU-hours. Sixteen source executions took 103 seconds and 54 evaluation executions took 273 seconds. The archived source collector revision was c45aa965db73ad0644382da578ff03a30b2083d7. The test donor collector was a8a997e7d06bb131d3f9710204c58e106f0f0904. Replay imported the existing ArmEnv checkout c30e7f76989563a808fc5985b0679d18a5066e4c and verified native LIBERO-Pro2.0 release e5500c690c1498eca1b7050350654964190cdf70. Plan SHA-256 before fit and test was 8941fd8ecb2ab54c880858b5e7569320c6b54477fd5dbed33ab03dba46607990. The parent recorded the registration, runner and initial scorer in 3927c398. The final scorer adds source-task and archived-reference sensitivity checks. Its executed SHA-256 was ee5ad9d0e8b27db0a392210ceac63e9f6e0c9f5d9f66a47538ef8ddc21b11766. The raw branch evidence is in this run directory. [Source receipt](evidence/fit/launch.txt) · [Linux logs and every native trace](evidence/traces.tar.gz)

On Baige, run from this directory with a fresh experiment path under the authorized scratch root. The first command runs the sixteen-replay source fit. The second requires its saved model. The third scores all matched branches. The completed remote scratch experiment was /tmp/frozen-adaptation-1124/timing-worker/cpu01.

```bash
trial=/tmp/frozen-adaptation-1124/timing-worker/repeat
bash cpu.sh fit "$trial"
bash cpu.sh test "$trial"
taskset -c 42 env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /mowenjie/fluffy-bassoon-2/setups/libero-meta-learning-20261117-r1/env/bin/python -B score.py "$trial"
```
