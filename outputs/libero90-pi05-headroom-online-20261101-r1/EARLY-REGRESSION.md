# Task44 early SAC regression — fixed experiment continues

- Motor: frozen official OpenPI40 pi0.5; task44, `turn on the stove`.
- Selection: native screen5/20; preferred task77 rejected on confirmation0/20;
  task44 separate confirmation5/20. All240 screens/40 confirmations retained.
- New-process controls beforeSGD: native6/20; unbounded tied Gaussian13/20;
  initialized bounded SAC8/20. Same fixed query bank, distinct sampling rules.
- First learned panel: **1/20 at20,580 actual updates**. Matched initials/seeds,
  fixed query weights, no replay insertion or query SGD; milestone hash checked.
- Later02:20 snapshot:32,520 updates,86 episodes,2,120 replay chunks;
  actor/critic losses3.1530/.04831 finite.51/86 training successes includes
  warmup15/22; subsequent learned-behavior cohort36/64 is a different state/time
  population and cannot be compared directly with heldout1/20.
- Whole-lineage actions143,916;6,786,232,643 output bytes. Latest recovery/raw
  hashes stable. Neither a finite loss nor a checkpoint establishes competence.

## Observed scalar changes, not a causal diagnosis

| TRAIN-minibatch statistic |180 updates|32,520 updates|
|---|---:|---:|
|Differential entropy|21.8539|-16.6251|
|Temperature|.948163|.0118874|
|Pre-tanh mean norm|.06569|7.01622|
|Average pre-tanh standard deviation|.872604|.551285|
|Marginal sampled bounded-action standard deviation|.586984|.760375|

The mean statistics are pre-tanh, with later extrema−3.113..3.023; bounded
sample extrema remain approximately±.999. These suggest more concentrated
conditional distributions centered toward action bounds. Marginal action
variance increased: this is not zero-noise or global variance collapse.
Negative differential entropy is valid; the released target is−16, so cooling
alone is not a temperature-sign bug. These32.5k TRAIN statistics do not describe
the20.58k query distribution and cannot establish why that panel failed.

Replay overfit/critic exploitation and a train/query generalization gap remain
plausible. Q(policy)=−2.955 versus replayQ=−3.918 is not proof of overestimation:
they predict different actions and are not observed heldout returns. Early
transient behavior remains possible. An independent narrow code comparison
found no concrete actor scaling, RGB/state order, reward/mask or entropy mismatch.

## Already planned discriminating evidence

Keep the fixed500k endpoint, every20k panel and final native control. The40k/
60k panels and existing entropy/temperature/mean/std/Q fields distinguish
rebound from persistent degradation. Compare incremental learned-behavior
training cohorts, excluding warmup. No extra physical panel, checkpoint/task
selection, active retuning or new learner seed is admitted by this diagnosis.
The untrained13/20 control already exceeds the native-only gain criterion;
native improvement alone cannot be called a learning benefit.

## Pinned recipe check

The upstream32D tanh actor is bounded to[-1,1], tiled without another gain;
the adapter preserves this. Single flipped64x64 agent-view RGB plus XYZ,
axis-angle and two gripper positions, query−1/success0, timeout bootstrap,
auto entropy−16, batch256/UTD20/warmup>500 match the pinned source. Deliberate
pi0→pi0.5 departures remain H50/execute20→H10/execute10, explicit reset banks/
settling and actual-duration discounting. These limit literal reproduction
claims; none establishes the cause of this regression.
[Pinned collector](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py),
[pinned learner](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/jaxrl2/agents/pixel_sac/pixel_sac_learner.py).

Evidence: CHECK-0220 and LEARNING-FIRST-UPDATES in the sibling runtime folder;
two independent static/scalar reviews. No raw arrays, checkpoint tensors,
model computation, tests or runtime observations were performed on the Mac.

## Planned40k panel rebounds

The40,140-update panel succeeds20/20 with1,651 primitives+200 settling,
versus the retained20,580-update1/20 and initial native6/tied13/SAC8. All20
initial identities/private seeds match native; acting weights stay fixed and
queries do not update or enter replay. This is a strong interim improvement
over every untrained control on the reused development bank. It weakens
persistent-collapse explanations without establishing an endpoint result.

The02:40 saved snapshot has53,120 updates/189 episodes,151 successes/38
failures. Between the two checks training adds100 successes and3 failures;
these repeated training cases still differ from the heldout query population.
TRAIN entropy remains near−16 (−16.520), temperature falls further to.00911,
pre-tanh mean norm remains6.622 and sampled bounded-action standard deviation
is.732. Thus concentration and low temperature coexist with the new excellent
performance; concentration alone does not explain the earlier failure.
Lower TD loss and a smaller actor-Q/replay-Q gap are compatible with improved
fit or replay coverage, not proof of critic accuracy or a causal mechanism.
The already-planned60k/80k panels test persistence; fixed500k, last-three
persistence and final native remain required. No active retuning or extra seed.
[Saved check](../libero90-pi05-headroom-runtime-20261101-r1/CHECK-0240.json).
