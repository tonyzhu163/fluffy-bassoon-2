# Rollout-rank RNG check

**The proposed identical-noise failure is not present in the checked finished
Drawer-to-Counter run.** All eight ranks have distinct saved CUDA RNG states,
distinct CUDA seeds and distinct actual initial-noise tensors. This holds for
both the qualification's first two training decisions and production's first
two training decisions. Python RNG states are also distinct across all ranks.

| Stage and record | Distinct CUDA states / ranks | Distinct initial-noise tensors / ranks |
| --- | ---: | ---: |
| q1-r1 `00000` (before-panel evaluation) | 8 / 8 | Not stored in the eval record |
| q1-r1 `00012`, first training decision | 8 / 8 | 8 / 8 |
| q1-r1 `00013`, next training decision | 8 / 8 | 8 / 8 |
| production-r1 `00000`, first training decision | 8 / 8 | 8 / 8 |
| production-r1 `00001`, next training decision | 8 / 8 | 8 / 8 |

The largest absolute pairwise correlation between the checked initial-noise
tensors is 0.016203. SDE innovations were recovered from the saved actual
next states, means and standard deviations. Restricting comparisons to pairs
with the same denoising index, their largest absolute correlation is 0.009840.
Identical underlying draws would have correlation near one; these values
provide no evidence of duplicated streams. The recovered innovations include
float32 state-rounding error and are not bitwise reconstructions of the
original Gaussian draws. These finite checks do not prove statistical
independence for every draw or cover every task.

The q1 first-training denoising indices by rank are `[5,8,9,0,2,7,2,5]`;
production's first indices are `[0,0,2,3,0,4,0,8]`. Repeated categorical indices
across some ranks are expected and do not imply shared RNG state. Each
production rank retains its own q1 CUDA seed, consistent with restoring that
rank's checkpoint RNG; seeds are distinct across ranks.

## Why the default-seed premise fails

The deployed HF worker's `init_worker` does not explicitly seed the sampler.
That does not make the implicit CUDA generator identical across processes.
In the exact installed PyTorch source revision
`70d99e998b4955e0049d13a98d77ae1b14db1f45` (PyTorch 2.11.0+cu128),
[`getDefaultCUDAGenerator`](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L50-L64)
calls `seed()` on first initialization, and
[`CUDAGeneratorImpl::seed`](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L320-L325)
obtains a nondeterministic seed. A separate explicit-generator factory in the
same source initializes a fixed seed; that is not the implicit-generator path
used by these `torch.randn(..., generator=None)` calls.

No `seed + rank` patch was made: the user's condition requiring matching rank
RNG states is false. Deterministic fresh-run seeding is a separate
reproducibility improvement, not a fix for duplicated exploration demonstrated
here. Existing checkpoint RNG streams and active worktrees were preserved.

## Evidence and scope

[RNG-RANK-AUDIT-Drawer.json](RNG-RANK-AUDIT-Drawer.json) records all tensor/RNG
hashes, individual seeds, pairwise correlations, and exact archive-member
paths, offsets and checksums. [RNG-RANK-AUDIT.py](RNG-RANK-AUDIT.py) is the
reproduction script. The audit read 40 records, about 74 MB, from the retained
native archive on the configured Baige Linux development endpoint. Every
selected record passed its recorded SHA256, and the archive manifest matched
the closeout receipt's SHA256. The full 100+ GB archive was not streamed again.
All tensor calculations ran on Linux CPU with zero model calls, new sampling,
simulator episodes or SGD. The report has 48 comparison rows because
production ordinal zero is also its first training decision.
