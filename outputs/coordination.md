# CTM in-context RL: coordination (live only for the CMM calibration)

Until the cutover, the live coordination file is fluffy-bassoon
[`outputs/coordination.md`](https://github.com/tonyzhu163/fluffy-bassoon/blob/lycoreco_rl/outputs/coordination.md)
on branch `lycoreco_rl`. Record no other status here yet; the CMM calibration
runs from this repository and is tracked below.

Cutover happens when the runs pinned to fluffy-bassoon commits (the Baige native
motor admission and gates, the Baige one-epoch CMM cost job and the compact
bandit T8 pair) have reported. Then copy that file here with its working notes
(`framing.md`, `candidate-experiments.md`), take a final archive snapshot in
fluffy-bassoon, and point fluffy-bassoon's file here. The plan is fluffy-bassoon
`outputs/lycoreco-rl-repo-split-20261028-r1/EXTRACTION.md`.

## CMM calibration (live)

Paper target (arXiv 2610.07907, few-shot linear regression, K=20): CMM
30.63 ± 0.24% query accuracy. Accuracy counts |error| < 0.05 on unit-variance
targets, so predicting the mean scores about 4%. Code:
`src/fluffy_bassoon/meta_rl/fewshot_regression.py`. Runs live under
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/{worktrees,outputs}/<run-id>/` on thu2.

| Run | State |
| --- | --- |
| Full calibration, CMM K=20 seed 0: thu2 job 48876, `fb2-cmm-cal-k20-s0-b6919d6-20261028-r1`, source `b6919d6`, node02 RTX 3090, user-approved | **Running, improving.** At chance through epoch 200 (MSE .907, accuracy 4.2%), then broke out: dev query MSE .714 / .444 / .143 / .096 / .091 and accuracy 4.6 / 6.1 / 11.7 / 14.8 / 15.4% at epochs 250 / 275 / 300 / 325 / 350. Last check 2026-10-28 11:22 UTC, update 58.5k of 157k; about 7.5 h left. Sync decay rates are still near 0 (max r .29 at update 57k), so the whole-episode average did not block learning. The user's rule was to cancel it if it had not improved; it improved, so it runs to completion. **Pass:** final 10,000-episode test within the paper's spread. Next: monitor; on completion read `run/summary.json` and report |
| Decay diagnostic: thu2 jobs 48983 (`fb2-cmm-decay-r1-k20-s0-bd38dd0-20261028-r2`, `--decay-init 1`) and 48984 (`…-u03-…-r2`, `--decay-init uniform:0,3`), source `bd38dd0` | **Queued (Priority), kept by the user.** User approval, verbatim: "Approved: implement --decay-init in fluffy-bassoon-2, run the CPU tests, and launch the two 50-epoch decay diagnostics (r0=1 and r0~U[0,3]) on thu2 under new run IDs. Leave 48876 running." CMM K=20 seed 0, 50 epochs, dev every 5 epochs, 3 h cap, RTX3090/4090D/6000D, `--exclude=node01,node02`. Jobs 48981/48982 were refused by the GPU probe because thu2 node02 GPU `fef80646` runs a foreign process outside Slurm; no training ran. Each step runs the probe, then the fewshot/CMM CPU tests (`tests.log`), then training. Question: does a higher decay start shorten the plateau? Compare dev MSE with 48876 (.9575 at epoch 25, .9565 at 50). Below 0.8 by epoch 50 means a speedup; report it, and ask before any full-length run with that init. Near .95 means no speedup within 50 epochs, not a failure |
