# Evaluation record

Record each result here with its source commit, cluster, job and receipt path.

## CPU test suite after the split, 2026-10-28

Secondary school cluster (thu2), job 48869, node01, 4 CPUs, 8 GiB, no GPU,
COMPLETED, exit 0. Source `9be85caf6e231a5bc5c59f03c77b484732b3e2eb` in a clean
pinned worktree; imports resolved to that worktree (`PYTHONPATH=src`, no
install). The full `tests/` suite ran in two existing environments:

| Environment | Python, torch, NumPy | Result |
| --- | --- | --- |
| `envs/lycoreco-de5277f-r1` (compact learners) | 3.12.14, 2.7.1+cu128, 2.2.6 | 203 passed, 17 skipped, 27 subtests passed, 93.5 s |
| `envs/rlinf-pi05-maniskill-e1a82eb-r1` (ManiSkill motor) | 3.11.16, 2.11.0+cu128, 1.26.4 | 203 passed, 17 skipped, 27 subtests passed, 119.2 s |

The 17 skips are the CUDA tests (`test_meta_rl_cuda.py` and two in
`test_meta_rl_ppo.py`), which need a probed GPU step. Receipts:
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/fb2-cpu-tests-9be85ca-20261028-r1/`
(job script, environment versions, pytest logs, JUnit XML).

## CUDA tests after the split, 2026-10-28

Secondary school cluster (thu2), job 48873, step 0, node02, one RTX 3090
(Slurm GPU 4, UUID `GPU-6e52b20d-e67e-9183-8251-f13388c512cd`), 4 CPUs,
16 GiB, COMPLETED in 26 s, exit 0. Same source `9be85ca` in a fresh pinned
worktree. The GPU probe ran inside the step and passed: one visible device, the
expected UUID through the GRES mapping, a CUDA allocation check, and no other
compute process on it before the tests. A foreign `VLLM::Worker_TP1` process
held 14 GiB on another node02 GPU outside Slurm; it did not touch this one.

With `FLUFFY_META_RL_CUDA_QUALIFICATION=1`, `tests/test_meta_rl_cuda.py` (GPU
rollouts match CPU and PPO updates for GRU, CTM control, CMM, looped
Transformer and Transformer-XL on both tasks; GPU checkpoint restore and the
next update) and the two CUDA cases in `tests/test_meta_rl_ppo.py` passed:

| Environment | torch, CUDA | Result |
| --- | --- | --- |
| `envs/lycoreco-de5277f-r1` | 2.7.1+cu128, 12.8 | 17 passed, 7.0 s |
| `envs/rlinf-pi05-maniskill-e1a82eb-r1` | 2.11.0+cu128, 12.8 | 17 passed, 6.5 s |

With the CPU suite above, all tests in this repository pass. Receipts:
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/fb2-cuda-tests-9be85ca-20261028-r1/`
(probe JSON, pytest logs, JUnit XML). A ManiSkill motor episode is still
unqualified in this repository.

## Prior evidence in fluffy-bassoon

All earlier compact results were produced by the same code in
fluffy-bassoon and are recorded in its
[meta_rl/EVALUATION.md](https://github.com/tonyzhu163/fluffy-bassoon/blob/0df7d026cb72d1d047bbb261ef95dc1f5d2da9fa/src/fluffy_bassoon/meta_rl/EVALUATION.md) at `0df7d026`. They are prior evidence, not
results of this repository; rerun anything the paper reports. The most relevant
sections:

- Environment and CMM recurrence; full-trial PPO and recovery (2026-10-24).
- Stochastic rewards and controller headroom; raw-clock Bernoulli learning
  screen; shared clock encoding and CUDA qualification.
- Reduced Copy memory qualification; full-length Copy preflight and CMM
  fidelity limit.
- Compact Transformer-XL control qualification.
- Completed five-core compact screen (GRU, CTM control, CMM port, looped
  Transformer, Transformer-XL on the six-arm Bernoulli bandit).
- Standalone core and native admission test cleanup (2026-10-28).

Two caveats carry over. CMM results from before the 2026-10-28 consolidation
used a different core and their checkpoints do not load at later sources. The
"CTM control" in those screens is the CMM with its memory blocks removed, not
the persistent CTM. The persistent-CTM bandit results (T1/T4 legacy and bounded
synchronization) are summarized in fluffy-bassoon `outputs/coordination.md`.
