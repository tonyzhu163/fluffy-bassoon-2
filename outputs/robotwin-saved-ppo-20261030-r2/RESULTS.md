# One real PPO update completed; optimizer setting is unstable

Baige job-yem1ctw27j26, source98efb377c525ff2c101b7e9ec0f6eaf115299b66,
exited0 and succeeded2026-10-30 02:11:35 Shanghai. Scheduler runtime32seconds.
All three original behavior likelihood/value replays were exactly0.0 after
restoring the motor's process-global high float32 matmul precision; default
highest did not reproduce them. No likelihood tolerance was relaxed.

The separate batch-three diagnostic reused12saved episodes/4,273source actions/
140noise decisions across phone_stand, object_stand and diverse_bottles. It
completed **one PPO update, five AdamW optimizer steps**, in5.580seconds;
full process9.620seconds. It issued **zero new robot actions**, did not advance
the batch-eight main comparison, and did not update SAC.

This is execution evidence, **not stable learning**. At3e-4/fiveepochs the
reported approximate KL was8.862768e19 and clip fraction0.7943. Gradient norm
before clipping averaged27.087. Weights changed and metrics remained finite,
but the distribution movement is unacceptable for launching a longer run as
configured. Preserve this bad-update checkpoint for diagnosis. The next bounded
source-only diagnostic inspects exact Gaussian KL, means/std/logratios and
recurrent reconstruction, alongside explicitly lower learning rates.

The18,788,171-byte full recovery (see exactsize inRECOVERY-AUDIT.json) passed
independent SHA/ZIP-CRC/CPUmetadata inspection:1update/5steps,63AdamW parameter
states each atstep5,12consumed episodes,0pending trials, complete optimizer,
sampler/schedule and globalPython/NumPy/Torch/CUDA RNG. The original checkpoint
SHA is unchanged. RecoverySHAc2f1fe3c7e21f1151ab453adab2e2520b5dafe2146936b8f31741091804bdcfb.

Reconciled actual usage after this job is43m42s of45minutes (78seconds left),
using the verified49second original startup rather than the earlier120second
reserve. This authorizes one last one-minute diagnostic within that ceiling,
not the pending two-hour source-learning allocation.
