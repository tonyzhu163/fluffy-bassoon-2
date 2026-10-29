# Initial submission: allocation-visibility preflight failure

Baige `job-c6u3tk8cf2y7`, submitted 2026-10-29 20:16:37 Asia/Shanghai,
source `a7aacdefc824d93fd188bc6750d674d2f757c38d`.

The allocation probe exited before all tests, model loading, simulator
episodes or training. Its NVML inventory contained one isolated GPU, but
`CUDA_VISIBLE_DEVICES` was unset. The launcher omitted the fallback already
used by this project's one-GPU Baige scripts; the probe correctly rejected
the missing visible-device list. No scientific result was produced.

Retry uses a fresh source commit and `robotwin-noise-matched-20261029-r2`.
Preserve any provided CUDA visibility; otherwise select the single
container-local device zero. The probe still independently attests allocation,
device count and CUDA computation. Reserve two minutes for this failed
preflight and cap the retry allocation at43 minutes, bounding both submissions
by45 GPU-minutes on scheduler caps. The learning contract stays unchanged.
