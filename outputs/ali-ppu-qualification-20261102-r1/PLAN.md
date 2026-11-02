# Ali PPU: offline workload qualification

Human request on 2026-11-02: test whether the Ali PPU resources can run
our tasks, using the existing `mwj_test` instance. Access guidance was read
from `/Users/mowenjie/fictional-chainsaw/AGENTS.md`; no historical study from
that repository is being resumed.

## Instance and current state

- PAI workspace 268463, region cn-wulanchabu.
- Existing user-owned DSW instance `mwj_test`, ID
  `dsw-q6smmq9wnvyu4n8y9x`, quota `DSW-PPU`.
- Existing configuration: one accelerator, 11 CPUs, 112 GiB RAM and SHM.
- Start request accepted; observed status **queued**, queue position 1.
  No active SSH endpoint or executed numerical test yet. The resource name
  does not establish the actual device or runtime.
- Existing image `archive/mwj:mwj`; inspect its installed framework before
  changing anything. Preserve the vendor PyTorch and accelerator libraries.

The old access guide's PPU qualification used DLC/gp7v. Whether this DSW
instance exposes a working PPU is an open question to resolve on the actual
allocated host. Do not submit a duplicate because it is queued, alter quota,
or touch other users' running instances.

**Startup outcome, Nov2 18:54:43 Shanghai:** the instance changed to Failed
after thirty minutes in the resource queue. The lifecycle table explicitly
reports `Resource queue timeout, lack of sufficient resources`; no working
SSH endpoint or numerical test became available. This is a capacity failure,
not a software/device compatibility result. See [startup receipt](STARTUP-RESULT.json).
Do not blindly restart the same request without a capacity change.

## Small test, once admitted

Use a fresh project-specific directory on verified persistent storage and
deploy the reviewed Git commit through Git/bundle. Record source, hostname,
Python/framework versions, actual device/memory, mount capacity and write
probe. Keep credentials and machine-local SSH configuration out of Git.

Run `QUALIFY.py` with the exact allocated hostname and a fresh output path,
bounded by a ten-minute process timeout. It requires Linux and one actual
working accelerator; CPU fallback cannot qualify accelerator support.

1. FP32 and BF16 matrix forward/backward against a CPU reference.
2. Actual repository history-policy AD/DPT supervised updates, with both the
   current decoder and legacy GTrXL, 2088 input features and six controls.
   Compare initial accelerator NLL with CPU, then verify finite gradients,
   updates and checkpoint/Adam/RNG continuation.
3. Actual residual PPO optimizer update on synthetic chronological data,
   including the existing KL diagnostics. Resolve small missing pure-Python
   dependencies in an isolated `uv` environment with vendor system packages;
   do not replace vendor torch/CUDA libraries.

This is a synthetic compatibility check: no simulator controls, real training
data, VLA weights, scientific policy training, task expansion or production
job migration. The script is prepared and syntax-checked on macOS, but not
yet executed or numerically qualified.

## Interpretation

Passing would support offline history-controller fitting and the residual
PPO update kernel on the tested stack. It would not establish full pi0.5,
RLinf/Ray/FSDP communication, production throughput, or robot performance.
Simulator stepping/rendering stays on Baige under the access guide. Do not
move active Flow SDE, Stage 0 or AD/DPT jobs. Treat a missing renderer or an
unsupported operator as a scoped capability result rather than changing
the scientific protocol to force a pass.

Save the actual result and logs here, record support and limitations, and
continue the existing queued instance until it is ready for this test.
