# Fast A1 cached rotation pilot

`python -m fluffy_bassoon.meta_rl.maniskill_rotation` is a CPU-only plumbing
pilot of the [A1 protocol](maniskill-pi05-icrl.md), using an existing motor
headroom cache. It reuses simulator outcomes; it launches no motor evaluation
and provides no live manipulation, held-out-object, held-out-angle or calibrated
architecture comparison result. Learners begin from random weights, so this
pilot omits the planned teacher warm start and its PPO anchor.

## Frozen task

Theta and correction phi both have the grid `[-30, 0, 30]` degrees. Their exact
sum uses only cached net rotations `[-60, -30, 0, 30, 60]`; there is no nearest
angle interpolation. A trial samples theta uniformly once and samples eight
distinct scenes without replacement from its split. Each action chooses phi
once, then samples a cached repeat uniformly for that scene and net angle.
Reward is the row's binary `success_once`. With one repeat, this is a
deterministic empirical outcome conditional on scene and net angle; resampling
scenes does not create additional independent simulator evidence.

The source headroom cache has 704 rows: 16 objects × 4 scenes × 11 angles,
one repeat each. Scene indices modulo four define the split: residues 0/1/2
are training (48 scenes), residue 3 is validation (16 scenes). Objects are
shared. These validation scenes also contributed to the original headroom
screen, so the split is for plumbing and development. Source episode identities
cannot cross splits, and every scene must contain all five required angles.
Rows are validated against a declared SHA256 and duplicate identities are
rejected. Grasp, distance, object name, instruction and other graded fields are
discarded by the cache loader.

Policy inputs are zero-valued physical state, previous correction one-hot,
previous reward, fixed attempt boundaries and normalized public clock. Theta,
scene IDs, repeat IDs and graded fields never enter observations or `info`.
One action ends each attempt; policy memory persists across attempts. PPO uses
whole trials with gamma and GAE lambda one. The learned curves use disjoint
environment reset seeds from training, with common seeds across conditions.

## Learners and controls

The CLI supports `persistent_ctm`, `transformer_xl` and `gru`. CTM uses bounded
synchronization and per-observation decay. Width defaults to 64; parameter count
and retained state bytes are reported. Equal width does not match capacity or
compute, so these first pilots cannot establish architectural superiority.

Every saved evaluation reports greedy and sampled behavior, each with normal
memory, memory wiped at every attempt, and reward input zeroed. Evaluation
restores the caller's Torch RNG. Paired reference strategies are zero correction,
the privileged angle-canceling oracle `phi=-theta`, and independent-arm Thompson
sampling with Beta(1,1) priors. The angle-canceling oracle is a physical reference,
not a strict upper bound on this small noisy empirical table. Both lesions may
have clock-dependent curves even without theta information; compare their
paired normal-condition gap and avoid claiming that any upward trend alone
establishes adaptation.

## Run and verify

Run CPU tests on allocated Linux resources:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src uv run --with pytest python -m pytest \
  tests/test_maniskill_rotation_poc.py -q
```

Use a clean pinned worktree and a unique output directory. A Slurm CPU step is
accepted directly; an exclusive allocated container requires its exact hostname:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ENV/bin/python \
  -m fluffy_bassoon.meta_rl.maniskill_rotation \
  --source-commit COMMIT --cache OUTCOMES.jsonl --cache-sha256 SHA256 \
  --expected-hostname ALLOCATED_HOST --core persistent_ctm \
  --updates 500 --num-envs 64 --eval-trials 128 --threads 2 --output-dir NEW_OUTPUT
```

Use separate output directories and training seeds for each core. Configuration,
per-update metrics, paired baseline arrays, six evaluation arrays and a complete
between-trial checkpoint are saved. Signals and the wall-time cap stop after a
complete update. `--resume PREVIOUS/latest.pt` restores optimizer and RNG state
into a new output identity, with an identical source/training/cache contract.
Any promising result needs additional training seeds, matched model budgets,
a separately collected cache and frozen live simulator evaluation under the
full A1 protocol.
