# Gate readiness readout

Flora is **not qualified for gate1 optimization**. A stricter terminal threshold
does not fix the retained error distribution. The frozen-motor learner and
common evaluation ports are implemented; native GPU qualification, temporal
reward qualification and new improving multi-task histories remain separate
work, not completed learning results.

Continue rubric iteration around the exact native predicates. This negative
threshold result does not establish that better wording cannot work. The human
clarification and a matched1–5 versus direct0/1 comparison are recorded in
[RUBRIC-NEXT.md](RUBRIC-NEXT.md).

## Existing-video threshold screen

All634 retained videos were analyzed on allocated Baige Linux CPUs, with no
model inference, new episodes or SGD. The384 TRAIN episodes were split by
native case into fit/internal validation;250 calibration episodes stay
separate by task. All videos have already informed development, so this is not
an unseen qualification. Original saved reward on the201 true successes and
183 failures gives:

| Success rule | True positives | False positives | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Hard Score5 |196|58|77.2%|97.5%|
| Expected-grade reward >=.80 |167|19|89.8%|83.1%|
| Expected-grade reward >=.90 |89|9|90.8%|44.3%|
| Expected-grade reward >=.95 |15|3|83.3%|7.5%|
| P(Score5) >=.95 |0|0|undefined|0%|

Eleven thresholds each for expected-grade reward and P5 were screened on
saved-original, rerun-original and revised-rubric scores. None met the proposed
fit criterion of precision lower95% >=.95, FPR upper95% <=.05 and recall >=.50.
Zero detections is not a successful repair. Full sweeps, both internal splits,
per-task calibration, confusion counts and Wilson intervals are in
[OFFLINE.json](OFFLINE.json). No production mapping was changed.

Terminal scores contain no causal first-success time. `flora_success.py`
defines threshold/hold decisions only from prefixes with an explicit last
visible control, reports the confirmation time rather than backdating it,
and separates early detections from never-success false positives. Holds over
chunk scores must not be described as consecutive-frame tests. Actual timing
precision/recall remains unmeasured until causal prefix scores and matching
native first-success traces exist at the target horizon.

## Motor noise versus selected case difficulty

The four completed, independently audited native8 banks contain80 selected
anchor failures at600 controls,640 episodes. No new simulator calls were made.

| Task,20 cases each | Fail all8 | Mixed | Succeed all8 | 3–5 successes/8 | Successes/160 | Oracle recoveries/20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenDrawer |1|19|0|12|79|19|
| DrawerToCounter |5|15|0|6|43|15|
| CoffeeSetupMug |4|16|0|8|44|16|
| TurnOffStove |11|9|0|2|16|9|
| Total |21|59|0|28|182|59|

OpenDrawer's unbiased within-case pair disagreement is44.5%; DrawerToCounter
31.3%, Coffee33.9%, Stove16.8%. All k/8 estimates, intervals and within/between-
case variance estimates are retained in OFFLINE.json. Eight observed failures
still allow a Wilson upper success probability of32.4%. These conditional
failure banks do not estimate full task competence or prove irreducibility.
Oracle best-of8 establishes resampling headroom; it does not qualify an actual
selector, noise-steering policy, useful experience or in-context adaptation.

## Implemented preparation

- `interface_ppo.py`: the joint policy action contains motor noise and steering;
  its full likelihood enters PPO. Public features precede sampling. Shared
  deterministic execution, gamma.99/lambda.95, first-success transitions,
  chronological AD targets and full optimizer/sampler/RNG recovery are exposed.
- `robocasa_interface.py`: pre-action prefix/proprio features, explicit native
  noise, frozen action/conditioning/velocity/noise/direct/selection maps and a
  first-success video/accounting environment. PPO and history deployment use
  the same raw action and map. Native GPU equivalence/throughput remain pending;
  CPU fixtures do not qualify the actual robot runtime. Direct and selector
  initializers must be qualified on their selected interface before training.
- `paired_evaluation.py`: one prospective schedule,200 cases/task by default,
  explicit draws and attempts, exact canonical pairing across arms/draws,
  first/cutoff success and case-cluster inference. `evaluation_actor()` binds
  the same checkpoint to full, memory-reset or reward-blind execution. The
  qualified simulator callback still owns physical/camera validation.
- `configs/native_ppo/gate4_history.json`:128-token training and model context,
  four source tasks, whole-stream validation and held-out-task folds. The
  existing one-task flat histories remain pipeline checks; no new fit or
  history collection was launched. New collection waits for improvement on
  the selected shared interface.
- FLORA-PPO now names gate0 agreement/headroom, gate1 the two reward controls,
  and gate4 meta-learning. Both AD-trend claims are explicitly exploratory;
  context extrapolation is an untested possible explanation for the OOD drop.

The first allocated CPU qualification passed36 tests at396b1065; the expanded
qualification passed60 at0af4705b, including the native binding and128-token
optimizer path. [Final source and command](CPU-FINAL.json).
The causal-prefix scorer subsequently passed8 allocated Linux CPU tests at
aad24b8d, including future-frame exclusion, prefix metadata and reader cleanup;
the legacy terminal path stays covered. [Scorer qualification](CPU-PREFIX.json).
The bounded manifest contains128 existing videos and1024 prefix requests;
preparing it performed no inference. [Requests](PREFIX-JOBS.json).
Evidence is retained alongside the source pin in this directory. No active
scientific worktree or running GPU contract was changed by this work.

The retained-monitor audit found437 files across the original/resumed Flora
runs, including excluded/partial-run records. None contains first_success_step,
success_steps or simulator_success_once. This includes the retained training
sidecars but does not turn437 files into437 admitted training episodes. Native
first-success timing for these old videos needs recovery/replay, alongside
new causal prefix scores. [Source-file audit](TIMING-AUDIT.json).

The next scientific step is continued native-criteria rubric iteration and
the matched graded/binary comparison in RUBRIC-NEXT.md. Thresholding did not
repair the reward, and prefix preparation is not a substitute for that semantic
work. No new-rubric, direct-binary or prefix scoring job was launched here.
