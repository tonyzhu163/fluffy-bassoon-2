# Atomic baseline verified; composite continuation closed

The user clarified that the current project stage trains the offline
history/steering modules on atomic tasks with pi0.5 frozen. Composite tasks
belong to later transfer evaluation. Extending the SFT reference check to all
50 tasks was unnecessary for this atomic-first stage; the continuation was
intentionally closed. No new learning allocation is planned by this run.

## Retained verified result

The complete18-task atomic evaluation yielded363/900=40.33%,against the
published39.6%. All cases,input/video hashes and native sampler counts were
independently audited. Original JAX Human300 step75000,pretrain,seed7,
prediction50/execution5/ODE10 and equivalent released horizons were preserved.
Zero training updates. [Atomic result](ATOMIC-RESULT.md).

The narrow1.0.1 release moves a1.5 horizon multiplier into the registry;
the original evaluator already applied it. The source audit found no change
to physics,controllers,reset or success predicates at that release boundary.
The matching atomic result provides no reason to downgrade the qualified
runtime for current work. [Research](RESEARCH.md).

## Intentional composite shutdown

Baige jobjob-omodtcqp0uzn received an operator-stop receipt at17:39:11 Shanghai
onNovember9. Its existing peer-failure cleanup stopped simulator clients and
closed all four native servers successfully,with final sampler counts and
JAX RNG keys saved on every rank. This receipt is explicitly an operator
control record,not a fabricated worker or benchmark result. The scheduler
labels the job Failed because intentional closure used that cleanup path.

All four pod allocations ended17:39:35; scheduler closure is17:39:54.
The independent [stop audit](COMPOSITES-STOP-AUDIT.json) verifies227 complete
diagnostic episodes,664,974 completed-case policy controls and133,000
completed-case predictions. Native servers processed138,941 predictions in
total; charging every returned prediction at5 controls gives a694,705-control
upper bound including interrupted episodes and unexecuted chunk tails.
All retained completed-case input/video hashes pass. Interrupted episodes
are excluded; this is not a1600-case benchmark score or an exact resumable
in-flight simulator state. Frozen original weights and all four native
sampler/RNG receipts are retained; there was no optimizer because no SGD ran.

## Cost and next scope

The closed continuation costs17,116 conservative GPU-seconds from creation
to scheduler closure,including queue delay. Actual pod-to-end allocation is
13,172 GPU-seconds. All attempts in this task total35,532 conservative
GPU-seconds=9.87GPUh,below50; active/reserved GPUs are now zero.
[Cost closeout](BUDGET-CLOSEOUT.json), [scheduler proof](COMPOSITES-CLOSED.jpg).

Keep pi0.5 frozen. Atomic source/meta-training and held-out atomic evaluation
must be separate; deployment adapts through completed history and derived
conditioning/KV activations at fixed weights. Further VLA fine-tuning,
OpenVLA migration and the proposed C5/C50 VLA-training comparison are
withdrawn. The atomic-first ICRL training contract is a subsequent task,
not an unlaunched continuation hidden in this SFT evaluation.
