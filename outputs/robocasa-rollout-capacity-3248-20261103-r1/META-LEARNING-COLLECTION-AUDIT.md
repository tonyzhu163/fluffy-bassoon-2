# PPO collection for four meta-learning methods

Human confirmation, 3 November 2026: the intended methods are AD, DPT, DIT
and value-based offline ICRL (SICQL/IQL style). This is a static source/schema
audit, not an actual completed PPO dataset or a new training launch.

Velocity and all-layer K/V are independent source experiments. Each needs its
own four method fits, using the same per-interface source episodes, splits,
reward convention and evaluation cases. Their output dimensions are 32 and
1152 respectively. Their histories and final teachers cannot be interchanged.
The 80 GPU-hour ceiling covers the two PPO sources and their qualification and
evaluation; this audit does not allocate downstream meta-training compute.

| Method | Source requirements | Current implementation |
| --- | --- | --- |
| AD | Actual actions/rewards in chronological behavior-policy order, episode boundaries and task-pure multi-episode streams | Producer and AD corpus/fit ports exist; synthetic allocated-Linux qualification is recorded. Actual revised 50-update/384-episode source remains unbound. |
| DPT | Same actual contexts, plus final same-interface teacher labels at each recorded query feature | Final-teacher relabeling and paired AD/DPT views exist. Final PPO policy is a candidate teacher; its competence is not established by being final. |
| DIT | Actual behavior actions and returns, in-context Q/V estimators and advantage-weighted action loss | Raw ingredients retained. No DIT exporter/trainer/qualification in the current steering path. |
| Value-based offline ICRL | Actual (feature, steering action, reward, next feature), consistent terminal/bootstrap masks and task-pure contexts | Raw observations/actions/rewards/boundaries retained. Next-feature/TD exporter, Q/V objectives and native evaluation integration still required. |

The DIT paper uses observed behavior-action labels with learned Q/V-based
weights ([§4.2 and Algorithm 1](https://arxiv.org/html/2601.20116v1#S4.SS2)).
The prior review's shorthand that DIT is weighted final-teacher DPT was
incorrect and has been corrected. S-ICQL uses offline transitions, Bellman Q
regression, upper-expectile V and weighted policy extraction; its complete
method also includes a learned world-model prompt
([§3](https://arxiv.org/html/2506.01299v3#S3)). A decoder with an in-context IQL
objective is an adaptation, not a claim of reproducing every S-ICQL component.

`joint_steering_ppo_worker.collect_bank` saves raw chunks before dispatch,
then adds observed rewards, next observations, actual mapped commands and
both logical and physical boundaries. It records current features, steering
samples/mean/std/log probability, critic values, task/case, behavior version,
native noise and source/runtime/interface identity. Episode checkpoints retain
all physical chunks, including post-first-success suffixes. The master saves
every pre-SGD bank and post-event actor/value/Adam/RNG recovery. None of these
may be pruned merely because public save/evaluation cadence is every five
updates. Final `history.npz` alone is insufficient for the four-method source.

`frozen_steering_history.save_corpus` currently emits AD/DPT train/validation
views: four files, representing two algorithms. `FIT.py` currently accepts
only `ad` and `dpt`. The producer-port checks are synthetic; no revised
scientific source exists yet. The actual source currently specifies simulator
first-success reward. It does not automatically provide Flora scores or a
Flora-trained corpus.

Before relying on the long collection, round-trip actual production episodes
through each intended consumer, including failures, early success, physical
timeout and at least two behavior versions. For the value consumers, verify
nonterminal next-feature joins; derive discounted returns and terminal masks
against a numerical reference. There is no explicit terminal next-feature in
the raw chunk; terminal bootstrap handling must be declared, with raw next
observations retained for any required re-encoding. Never join across resets.
Choose logical first-success or physical-end value semantics explicitly;
do not silently mix the PPO prefix with the full-history suffix. Offline
return supervision must not become future-reward leakage into actor inputs.

Each arm targets 19,200 episodes, 3,840 per training task, and 230,400 H50
steering decisions (11.52 million primitive controls) across 50 updates.
Physical action count alone is not a guarantee of useful meta-learning data.
AD needs measured source improvement; DPT needs a qualified teacher; the value
methods need action/reward coverage and stable value fitting. All methods need
task-pure splits and held-out evaluation. Five source task families and one
learning run per interface do not establish broad task generalization.

Conclusion: the planned raw collection is compatible with all four in
principle, provided the stated retention and boundary conventions are kept.
Only the AD/DPT pipeline is currently implemented and synthetically qualified;
claiming all four are already ready would be unsupported. No extra simulator
collection should be needed solely to obtain the missing schema fields, but
actual consumer round-trip qualification is still pending.
