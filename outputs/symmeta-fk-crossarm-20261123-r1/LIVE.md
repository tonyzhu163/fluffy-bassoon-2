# Three-panel paired native evaluation

Each pair uses the same frozen motor, released query, per-chunk noise bank and full-precision seed-41 gain and grip. The intervention replaces the estimated yaw with the measured FK yaw. On π₀, both arms use the historical quaternion sign rule on every motor observation. Only verified pairs contribute to success rates.

## Sources and qualification

| Panel | Frozen motor | Historical source commit | Baige archive |
| --- | --- | --- | --- |
| IIWA | π₀.₅ | `490f25c5716aa067199c5b65e32caf6cd15421a4` | `gpu09iiwa` |
| Kinova3 | π₀.₅ | `465f33dc657ceab341dbceaf6b98355b94810059` | `gpu06kinova3` |
| UR5e | π₀ | `490f25c5716aa067199c5b65e32caf6cd15421a4` | `gpu08pi0fix` |

I checked the remote clean HEADs, launch/config receipts, evaluator and motor signatures, and the reset lines in all 24 original shard logs via read-only SSH. The source checkouts are `source-wave2b` (Kinova3) and `source-wave2d` (IIWA and UR5e) under `/mowenjie/fluffy-bassoon-2/setups/xemb-icml-campaign-20261121-r1/`. The native launcher, evaluator, weights manifest and individual historical shard hashes are recorded in [QUERY_CONFIG.json](QUERY_CONFIG.json). The original pairing and EGL procedure is documented in the [UR5e launch record](../symmeta-fk-ur5e-20261123-r1/LAUNCH.md).

On Baige, the evidence capture point for each original panel is `/mowenjie/fluffy-bassoon-2/outputs/xemb-icml-campaign-20261121-r1/PANEL/`. The runner verifies a panel-specific SHA-256 over the exact bytes of launch.txt, config.txt, the eight live-shard.json files, the eight support-meta.json files and the eight original query-reset log lines, in that order and in shard order within each group. It verifies the active and passive tapes individually, the mounted-arm identity, seed-41 checkpoint and motor weight files. Every row joins by ordered index, task, transform, support physics, tape paths and digests, seeds, layouts and original native record.

The final `cases.json` schema records exact CPU agreement with the historical seed-41 loader/predictor and the separate residual from rounded historical GPU output. The first CPU attempt's rounding comparison cannot supply a qualified case file. On the CPU02 UR5e π₀ export, all 96 inputs qualify and its raw JSON SHA-256 is `278a291882ca5e7252b4beb36f6c2e2d7a69bd2a283bba0ec9d8c30b8741b053`. The offline worker recorded active-replay failures on IIWA and Kinova3. Use its final per-case availability and readback for each panel. [Qualification results](RESULT.md) · [Reader](qualify.py) · [FK and reset receipts](fk.py)

## Parent invocation

The qualification worker's [evidence manifest](evidence/summary.json) pins each raw Baige cases.json by run, revision and SHA-256. Its three final exports contain 84 eligible IIWA pairs, 84 Kinova3 pairs and 96 UR5e π₀ pairs. IIWA indices 48–59 and Kinova3 indices 84–95 have null FK and are withheld. The manifest locks all 288 source cases, including those 24 exclusions. A completed evaluation on this manifest covers 264 pairs (528 native episodes).

From the parent-deployed, clean pinned checkout in one exclusive Baige Distributed Training container, set `PIN` to its verified full Git SHA. This invocation uses the exact qualification manifest checked on November 23, eight qualified RTX PRO 6000D cards and a one-hour allocation.

```bash
RUN=outputs/symmeta-fk-crossarm-20261123-r1
bash "$RUN/live.sh" "${PIN:?deployment SHA}" live01 8 3600 \
  "$PWD/$RUN/evidence/summary.json" \
  d86d8fe684dd57a6f895bccd76d34a582a22a55eaccaee0af70cea27f5d66719
```

A parent-supplied manifest may instead map the same three panel names to absolute raw `path` and independently reviewed `sha256` values. That form requires all 96 FK estimates unless the parent explicitly adds an exact ordered `withheld_indices` list for the affected panel. Pass that manifest's absolute path and SHA-256 in the last two argument positions.

The arguments after the attempt name are allocated GPU count (1–8) and total wall seconds. Use a fresh name for each attempt. The launcher caps each invocation at eight GPU-hours and includes preflight, all policy loads, rollout and merge in its deadline. Match the scheduler wall limit and disable retries. Completion time on these panels has yet to be measured. The parent tracks cumulative allocation across attempts.

Prepare verifies all three panels before scoring. Device lane `d` takes tasks `d, d + count, … < 24` in order, with eight shards each for IIWA, Kinova3 and UR5e. Each nonempty shard starts a new process using that panel's historical source and one frozen policy. One unscored create verifies the model and NVIDIA renderer. Each scored arm has up to three pre-inference creates on the official pairing callback. A completed trajectory is never retried. Both arms must match the entire fresh reset, noise and opening proposal. Native prompt, renderer, action schema, noise digest, model class and horizon must also match the historical query receipt.

## Outputs and failure interpretation

The new attempt lives at `/mowenjie/fluffy-bassoon-2/outputs/symmeta-fk-crossarm-20261123-r1/ATTEMPT/`. Its manifest copy, config, GPU probe and per-panel prepare logs identify the run. Each `PANEL/panel.json` locks the joined inputs and historical reset receipts. Each atomic `PANEL/shard-N/II.json` retains the two full native records, step rewards, return, success, terminal flags, control counts and opening-proposal digests.

Each `PANEL/summary.json` requires every eligible pair in that manifest. It reports overall, OOD and task-level successes, FK wins/losses/ties, both-failed indices and native success differences with task-cluster bootstrap intervals. Historical successes and changed baseline outcomes are reported separately. For partial panels the estimates describe the available subset. The original full-panel denominator is 96.

`status.json` records withheld indices, missing eligible indices, completed pairs and failed stages separately for each panel. A failed or timed-out attempt has a nonzero `exit.txt`. Shard logs include the native reset differences on rejected creates. If the scheduler externally kills the entire allocation before the exit handler runs, use the atomic pair records, per-stage logs and return codes for recovery.

Local static checks passed under Bash and Python 3.11. The evidence readback verified the three raw JSON digests, all 288 reader residuals, every available planar FK estimate, and exact 24-task coverage for each GPU count from one through eight. Fresh native outcomes await the parent launch.
