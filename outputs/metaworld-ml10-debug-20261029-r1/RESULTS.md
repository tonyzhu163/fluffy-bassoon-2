# ML10 debugging results and setting decisions

The strongest confirmed issue is public-clock extrapolation in long evaluations, especially for CMM. The measured critic-gradient imbalance is real, but lowering CTM's value-loss weight did not improve the prespecified learning endpoint. These checks support keeping T1 and the current learner settings while treating clock extrapolation as a separate evaluation question. They do not establish a CTM/CMM advantage.

## What happened on this bench

Original training uses 50 balanced lanes, five500-step episodes per trial, ten PPO epochs and80collections/10Mtraining transitions, seed1000. Original evaluation uses25episodes with the training clock scale: the episode input ranges0to1 during training, then grows to6 during evaluation. A policy can therefore lose performance because of an unseen input range. Training and held-out families also need separate reporting; the goal-visible V3/resampled-instance protocol is not a METTA reproduction.

The low CTM training score is not solely a long-evaluation artifact. Its first-five-episode fresh train-family success is16%, versus RL²28% and CMM24%. ECET's current first-five snapshot is38% at30/80collections (3.75Msteps), not a terminal10M result; AMAGO finishes at10%. Sampled training-rollout success over the last ten collections averages CTM20.4%, CMM23.84%, RL²29.08%. These use training pools and sampled actions, so they are descriptive checks, not controlled evaluation comparisons.

[First-five-episode companion chart](first5-success-vs-active-hours.png) independently recomputes every plotted prefix from raw family-balanced outcomes. All originally incurred training/evaluation time remains charged, including fullK25evaluations. One seed; no SEM or extrapolation. Original25episode comparison curves remain unchanged.

## Frozen clock intervention

Each condition uses the original10Mpolicy with frozen weights and normalizer, retains original deterministic actions, ten balanced lanes/K25/H500 and the same pool hashes/seeds, and changes only the normalized episode clock by capping it at1. Each clock condition is crossed with normal versus episode-reset memory. This is separate diagnostic evidence, not a replacement headline score.

| Policy | Split | Memory | Original clock | Capped clock |
| --- | --- | --- | ---: | ---: |
| CTM | Train families | Normal | 16.0% | 18.4% |
| CTM | Train families | Episode reset | 18.4% | 18.8% |
| CTM | Held-out families | Normal | 0.0% | 1.6% |
| CTM | Held-out families | Episode reset | 0.0% | 0.0% |
| CMM | Train families | Normal | 10.0% | 20.8% |
| CMM | Train families | Episode reset | 22.8% | 24.8% |
| CMM | Held-out families | Normal | 22.4% | 20.8% |
| CMM | Held-out families | Episode reset | 26.4% | 39.6% |

CMM's normal train score over the first five episodes stays24% under both clocks; its later20episodes improve6.5to20.0%. This explains much of its apparent long-trial decline. Improvements mainly concern drawer-close and reach; many training families still have zero success. CMM's held-out normal score does not improve. The high39.6% held-out reset/capped score mostly comes from door-close and drawer-open; drawer-open rises32to98% within that family. Persistent memory remains weaker than reset, so these scores do not establish in-context learning or a generalization advantage attributable to memory.

CTM's capped normal score improves only2.4percentage points, leaving the original low learning performance largely unresolved. Its small held-out1.6% result is not a reliable generalization claim. Increasing T is not a demonstrated remedy.

Both jobs passed8allocated qualification tests, including actual model prefix/no-op checks forGRU/CTM/CMM and a realML10simulator trial. Each completed all8conditions/1Mevaluation transitions with zero training or optimizer steps. All four complete first-five-episode trajectory hashes match within each clock pair. Every original outcome/return/family array reproduces the original checkpoint's final evaluation exactly; all model/normalizer fingerprints stayed frozen and parent checkpoint hashes were independently rechecked. Saved-artifact audits independently recomputed success, per-episode curves and family weighting for all16clock conditions.

- CTM: Baige `job-vb7p9f0cz56n`, source`b790031791acf26309a5e6fcafa465e4a331e51f`,584launcher-active seconds. [Audit](../metaworld-clock-debug-baige-20261029-r3/audit.json), [curves](../metaworld-clock-debug-baige-20261029-r3/frozen-clock-success.png).
- CMM: Baige `job-pgjw4p0j22ut`, source`a95271a453d70f3b965da7b0979cc1a623000c6a`,506seconds. [Audit](../metaworld-clock-debug-baige-20261029-r1/audit.json), [curves](../metaworld-clock-debug-baige-20261029-r1/frozen-clock-success.png).

## PPO mechanisms and the controlled learning result

On an actual25kminibatch from each completed checkpoint, the shared-core weighted critic/actor gradient norm ratio was CTM142 and CMM131.3. Synchronization decay rates are learned and sampling standard deviations remain broad; neither all-zero decay nor collapsed Gaussian variance explains the results. A gradient ratio is not an Adam-step ratio, and global clipping does not multiply Adam step sizes by its scalar.

The cached100TBPTT implementation reuses behavior-policy boundary states. Before optimization, full-prefix/cached forward replay matches within measured FP32 roundoff. After ten PPO epochs, cached-versus-fresh saved-action logprob RMS differs by CTM.03564/CMM.04468. This measures a recurrent approximation, not distribution KL and not yet a causal learning bottleneck. Reducing minibatch_trials10to2, on the identical125krollout and1.25Mloss samples, raises Adam steps50to250, increases drift to.07449/.10835 and slows PPO38.23to182.07s /30.44to149.00s. Do not shrink minibatches alone as a speed or fidelity fix.

A paired CTM learning control then forked the same complete10Mpolicy, Adam moments, normalizer, sampler and all RNG streams. Only value_coef.5versus.005 changed. Each serial branch collected1.25Mnew transitions over ten rounds and used500Adam steps/12.5Mloss forwards. The first baseline rollout, policy and full Adam state exactly reproduced an independent diagnostic checkpoint. Both final iteration90/11.25M/4500Adam checkpoints contain full recovery state and match their saved hashes.

| Final metric, average of two paired evaluation blocks | Value weight0.5 | Value weight0.005 |
| --- | ---: | ---: |
| Train normal |16.2%|16.0%|
| Train episode reset |17.4%|17.0%|
| Held-out normal |0.8%|2.0%|
| Held-out episode reset |0.4%|1.2%|

The prespecified train-success endpoint does not improve. The lower coefficient reduces the initial gradient ratio142to1.42 but increases first-update cache drift.03564to.04899. Small held-out counts do not justify selecting it for generalization. This is a short warm continuation with inherited Adam moments; it does not rule out a different coefficient in training from scratch. All16raw evaluation conditions and two recovery hashes were independently audited. Both branches retain the original training seed; the two evaluation blocks are not training-seed replication.

Control: Baige `job-knyy09k3p6rr`, source`08a9700616bb4cbac70080f7d3462b03bd1bdd53`,2430launcher-active seconds. [Audit](../metaworld-ppo-controls-baige-20261029-r1/audit.json), [checkpoint audit](../metaworld-ppo-controls-baige-20261029-r1/checkpoint-audit.json), [curves](../metaworld-ppo-controls-baige-20261029-r1/value-control-success.png).

CMM received its own optimizer diagnostic (`job-eouydzi8z936`, source`0b05ddf`,78tests/291seconds), not an assumed transfer of CTM measurements. [Audit](../metaworld-ppo-debug-baige-20261029-r2/audit.json), [recovery hashes](../metaworld-ppo-debug-baige-20261029-r2/checkpoint-audit.json). There is no evidence here for adopting a lower CMM value coefficient; its prepared control launcher was not run or promoted.

## Decisions

1. **Keep T1, value_coef0.5 and minibatch_trials10 for the current learner.** No tested change shows that increasing internal iterations, shrinking minibatches or lowering the value weight repairs learning. T2/T4, depth and trace length were not ablated; their optimum remains unknown.
2. **Separate competence within the trained five-episode range from25episode extrapolation.** Report the first-five companion for every method. For future long-horizon claims, predeclare a bounded/absent/randomized clock protocol and apply it consistently across methods. The validated cap is an evaluation intervention; silently substituting capped CTM/CMM scores into the original comparison would be invalid.
3. **Keep reset-memory controls central.** CMM's improved capped scores still favor reset; do not present them as learned history use or VLA admission.
4. **If further training is proposed, prioritize recurrent PPO fidelity and reward/value scaling over a broad architecture sweep.** Refreshing cached states with the current policy and testing a longer gradient horizon are concrete untested candidates. Family reward/advantage scales vary sharply, so global value explained variance can obscure poor within-family prediction. These require new controlled experiments; none was auto-launched or presented as a proven fix.

The code added an evaluation-only clock adapter/driver, qualification tests and immutable source-guarded launchers. Training model/PPO/environment code remains byte-identical to each qualified parent. It does not change active/default training contracts. All model execution/tests ran on allocatedLinux; macOS/dev work read saved artifacts, edited code, parsed metadata and rendered plots.

## Provenance, costs and retained failures

Original CTM/RL²/AMAGO completed10M, while ECET remains active under its existing monitor. The debug investigation is complete independently of that unfinished comparison. CMM's original job and paper-calibration campaign have separate ownership; their sources/checkpoints/jobs are unchanged. Paper-level CMM calibration is not established by this ML10 audit.

CTM diagnostic r1 retained its failed too-tight absolute value tolerance; corrected scale-aware checks accept small FP32 roundoff while rejecting material mismatch. Its identical raw rollout, complete diagnostic checkpoints and full failure receipt remain saved. CTM clock r2 (`job-l6xdatciygju`) failed before evaluation because its test called a CMM CLI preset missing in the old pinned runner;6checks passed/2failed. Corrected r3 supplies exactly the same constructor configuration explicitly, retains all8cases and changes no model code. The failed25launcher seconds (71scheduler-reported running seconds) are recorded separately. No failed work was overwritten.

One GPU per method throughout; serial branches and diagnostics; no new training seed, full sweep, VLA work or expanded cap. Checkpoints retain policy/optimizer/sampler/normalizer/Python/NumPy/Torch/CUDA recovery. Driver/launcher active costs are measured wall time on one allocated card, including CPU simulation/evaluation; they are not scheduler billing estimates.

Cumulative recorded CTM parent+failed/qualified diagnostics+value control+clock: **10797s/2.9992active GPUh**, below original6h. CMM parent+diagnostic+clock: **7363s/2.0453GPUh**, below original3h. Failed r1 uses its131second scheduler-pod receipt because its old launcher lacked a failure-time counter. The additional46seconds between clock-r2 launcher and scheduler run reports also leave substantial headroom.

Final evidence is mirrored locally and small receipts/plots on shared storage; original raw rollouts/checkpoints remain on shared storage. No debug jobs remain running. The existing20minute original-method monitor continues for ECET; it must not launch tuning or duplicate terminal work.
