# Fixed5,000 wide-head cached-plan diagnosis

The optional report ran after the required54 CPU cases in the same allocated
job-4jq7u25x1sv6. Both components passed,85/180 inclusive CPU-worker-seconds,
zero GPU/actions/SGD. Script4084144 and endpoint19ed635c were fixed before the
report. This reads all90 cached plans from whole8 TRAIN/4 VAL, including failed04.
No VLA, simulator or new robot query ran. Root read saved scalar outputs only.
[Exact report](../archive/command-readout-linked-resource-cpu-20261101-r1/wide5000-analysis.json).

| Cached submitted-plan measure | Native proposal | Wide5,000 |
| --- | ---: | ---: |
| TRAIN raw14D MSE |0.00942534|0.00547007|
| TRAIN arm12 MSE |0.00434094|0.00469944|
| TRAIN raw grip2 MSE |0.03993177|0.01009388|
| TRAIN actuator-clipped grip2 MSE |0.03991332|0.00832521|
| VAL raw14D MSE |0.01016262|0.03020118|
| VAL arm12 MSE |0.00486041|0.01022310|
| VAL raw grip2 MSE |0.04197587|0.15006966|
| VAL actuator-clipped grip2 MSE |0.04192132|0.14885749|

The aggregate training improvement comes from gripper coordinates; arm error
is slightly higher. Validation worsens in both arms and grippers. Native
actuator clipping barely reduces the validation grip error, so raw overshoot
alone does not explain the observed validation result. Phone09 and11 remain
large after clipping: grip MSE0.291787/0.268985 versus native0.041410/0.020735.
Phone08 and10 improve aggregate raw plan error, so validation cases differ.
Failed04's plan error improves, but it remains a failed historical rollout;
this analysis supplies no rescue or competent-teacher result.

Every reported independent per-slot interval floor is zero. This rules out
simple coordinate-envelope infeasibility on these submitted labels; it does
not prove network capacity, optimization success or physical executability.
The CPU reconstruction differs from the published GPU TRAIN/VAL aggregates
by3.59e-7/4.08e-6, within the predeclared roundoff tolerance. A changed payload
schema or large reconstruction mismatch is not observed here.

Each VAL case has224 submitted plan slots and193 historically executed
commands. Both sides of the clipped-grip metric are clamped to[0,1], so the
31 tail slots can contribute at most31/224 (about0.1384) to an average error
difference. The09/11 clipped-grip degradation exceeds that bound. Tail slots
alone therefore cannot explain their offline regression. This is a bound
on the cached metric, not physical independence: native planning consumes
the full submitted chunk before execution.

The action loss also uses one captured native base/noise per observation;
KV fitting uses resampled flow corruption. Scene transfer and decoder-noise
transfer are not separated by this bank. This is a remaining data-axis
confound, not an established cause or an admitted augmentation experiment.

Generalization failure/limited whole-case diversity, optimization and head
capacity remain competing explanations. These data do not choose among them.
The next discriminator remains the fixed final5,000 native/wide/KV robot
comparison; no validation-selected checkpoint, filtered case or new fit is
admitted by this report. The first readout dispatch failed before model/env
construction in approved-callable provenance lookup,0 episodes/actions and75
GPU-seconds. A narrow implementation repair keeps the original15/6,000/1,500
contract and remaining1,425 seconds. This report never gates that repair.
