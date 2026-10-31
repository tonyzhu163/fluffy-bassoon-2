# GPU admission stopped before training

Baige job `job-ikplxyvyh7p9`, submitted 2026-10-28 19:17:24 Shanghai, source
`5c91eb578e3aee1ff067fd4ead59d8d490298c34`, one NVIDIA RPBZZZ6, 16 CPUs,
64 GiB RAM. The exact clean source was fetched through Git into the pinned
worktree. Runtime: Python 3.11.17, torch 2.11.0+cu128, NumPy 2.4.6,
Meta-World 3.1.1, MuJoCo 3.3.0 and Gymnasium 1.4.0.

At 19:17:33 Shanghai the GPU probe rejected missing `CUDA_VISIBLE_DEVICES`.
Baige set `NVIDIA_VISIBLE_DEVICES` but left the CUDA selector unset. The
launcher did not supply the explicit single-device selector required by the
repository probe. No model tests, simulator trials or training ran. The
launcher saved exit code 1 and a failed GPU-probe receipt.

Correction: initialize an unset CUDA selector to container-local ordinal zero,
retaining the scheduler's NVIDIA allocation. The same allocation count and
PCI/PID checks remain mandatory. Retry in r2, with a new pinned source and
fresh output identity; do not alter the r1 worktree.

Raw receipts: `/mowenjie/fluffy-bassoon-2/outputs/metaworld-ctm-pilot-baige-20261028-r1/`.
