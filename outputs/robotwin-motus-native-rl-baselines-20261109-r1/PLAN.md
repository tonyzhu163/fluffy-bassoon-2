# Native Motus PPO and GRPO baselines

The prerequisite SFT readout is audited: phone stand27/100 clean,25/100
randomized (source1c7cc418 records the closeout). It is a documented
reconstruction, not a reproduction of Motus Table14's49/53. Preserve that
limitation in all subsequent comparisons. Do not tune on the200 frozen cases.

First qualify the actual RLinf RL model, with and without a PPO value head,
against the direct reference on the first8 lossless initial inputs from each
domain, selected by index before model output. Use identical explicit FP32
noise, B1, H32/K10, original normalization and measured state. Input transforms
must match exactly; decoded actions must have maximum absolute error <=1e-3.
Save both action arrays, preprocessing comparisons, frozen checks and source
hashes. No optimizer update or simulator episode enters this numerical check.
Failure holds training and triggers diagnosis; do not loosen the threshold.

Qualify transforms/imports on Linux CPU first. Reserve1,200GPU-seconds for a
single20-minute one-GPU numerical job (worker15minutes), within the same
100GPU-hour Epoch2 cap. Actual allocation and CUDA checks run inside the job.
The existing native renderer/simulator qualification remains unchanged; this
job makes no new environment-equivalence claim.

After numerical parity, qualify an RL environment bridge using the exact dated
native controller, all per-target observations and400 delivered targets,
including the partial final16 targets of the thirteenth H32 chunk. Bind the same
camera/state/instruction interface. The old fast sparse controller, modified
waypoint fallback, H384 budget and intermediate-render skipping are not the
measured SFT protocol and must not be inherited silently.

Then start independent fresh PPO and GRPO lineages from the same released SFT
weights, frozen VLM/action-expert learning, native simulator reward, shared
training cases excluding the200 evaluation cases, complete checkpoint recovery,
and matching evaluation inputs/cases/protocol. Retain existing qualified native
RLinf objective/replay/value checks. Measure collection/update cost before
reserving longer training. No history/student work or six-task expansion precedes
the phone baselines. Detailed training batch/schedule will be fixed after the
native bridge and measured capacity pass; no old stopped job is resumed.

## User steering: SFT initializer selection

The user raised similar public concerns about released Motus weight identity and requested RoboTwin SFT weights. The prepared Motus parity job was never deployed or launched; no CPU model construction, GPU computation or training occurred in this phase. Hold it while selecting a clearly documented SFT initializer. No paper/checkpoint mismatch is established by the public comments reviewed so far. The measured Motus27/25result remains a result for its exact pinned checkpoint/runtime, not an assertion about the paper weights.
