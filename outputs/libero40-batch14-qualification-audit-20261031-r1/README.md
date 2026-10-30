# Native batch14 PPO qualification: complete

Job-ccniwh46yq9w, exactsource9bc9c9c5d3a41cfb0be361d47ea3c4fa58a3d1a9.
Exit0;35CPU checks+2actualOpenPI motor-RNG checks and GPU probe passed.
Fourteen complete K4 trials/56successful episodes/795real decisions, one
global batch/two Adam steps,8,313all-in actions (7,743primitive+570settling).
No failure/recovery example or target performance is established.

Exact individual/merged log-probability and value replay errors areall0.
The candidate matches all66state tensors in the full recovery; allfinite.
Adam has63state entries atstep2; sampler, Python/NumPy/Torch/CUDA RNG, typed
motor RNG and source cursor are present. Full recovery SHA45888e863fce5a68b0fe18d4b6d3c0352e556c735debb154f43a726673200e83.
Actual motor parameter hashc565dd3f… unchanged before/after.

852conservative GPU-seconds (UI811). Collection651.24s, replay4.09s,
update13.46s; sampled whole-device peak10,804MiB, Torch update peak
1,345,175,040bytes. This is one simulator at a time with a14-trial update
batch, not14parallel lanes or a proven packed speedup.

Steady collection+replay+update extrapolates roughly22.4GPUh/Mactions;
including this cold setup gives28.5GPUh/M. Both are single-batch estimates,
not a full40-case distribution measurement or guarantee. Preserve additional
reset/qualification/retry costs in a later capped production plan.

receipt.json records the runner's final measurements plus a separate CPU-only
checkpoint tensor/hash read. No additional model forward/gradient/test was
run during that audit. This14-goal qualifier is not admitted asall40trained.
