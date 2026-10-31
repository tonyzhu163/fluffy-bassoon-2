# Learning to Steer: In-Context Reinforcement Learning for Frozen VLAs

Can a history-conditioned steering policy adapt a frozen vision-language-action
model to held-out robotic tasks with high sample efficiency and strong final
performance? We start with LIBERO and compare against ordinary DSRL.

[Paper draft](docs/paper/DRAFT.md) · [Research decisions](docs/paper/DECISIONS.md) ·
[Verified LIBERO results](docs/frozen-vla-libero/EVALUATION.md) ·
[Frozen corpus split](outputs/libero-pi05-corpus-20261029-r1/PLAN.md)

Earlier work asks whether a Continuous Thought Machine (CTM) can do in-context
reinforcement learning.
This repository holds the trial environments, the full-trial recurrent PPO
trainer, the matched learners (persistent CTM, Continuous Memory Machine,
Transformer-XL, GRU, looped Transformer), the CMM calibration task and the
frozen pi0.5 ManiSkill motor used for the robot demonstration.

It was split from `tonyzhu163/fluffy-bassoon` (branch `lycoreco_rl`) on
2026-10-28; [PROVENANCE.md](PROVENANCE.md) lists every copied file and its
source hash. Agents start with [AGENTS.md](AGENTS.md) and the live
[coordination file](outputs/coordination.md).

## ML10 results

There are two comparisons, each at 10M training transitions per run:

| Comparison | Policies | What is matched | Plot coverage |
| --- | --- | --- | --- |
| [Common-PPO core controls](outputs/metta-common-ppo-chart-20261029-r1/README.md) | GRU, Transformer-XL, CTM, CMM | Common PPO setup and 250k parameter cap; actual counts 232,361–249,095 | Final 10M-step endpoints; GRU/XL have no intermediate evaluations |
| [Native-method comparison](outputs/metaworld-ml10-matched-baige-20261028-r1/README.md) | CTM, RL²-PPO, ECET, AMAGO1 | Training interactions; model sizes and learners differ | Learning curves evaluated every 625k training steps |

Both use seed 1000. The CTM checkpoint is shared across the comparisons;
these are not two independent CTM runs. Neither establishes a CTM advantage.

## Layout

| Path | Contents |
| --- | --- |
| `src/fluffy_bassoon/meta_rl/` | Trial environments, PPO, training and evaluation CLIs, Transformer-XL, CMM wrapper, Copy and few-shot regression tasks |
| `src/fluffy_bassoon/rl/` | ManiSkill pi0.5 motor settings and rollout, RLinf checkpoint compatibility |
| `src/lerobot_policy_ctm/` | Recurrent cores: persistent CTM, CMM, looped Transformer action experts, Sakana CTM reference components |
| `src/fluffy_bassoon/{gpu_execution,baige_gpu,evaluation_stages}.py` | Execution-mode checks and receipt helpers |
| `scripts/` | GPU probe and availability check, ManiSkill gate and headroom scripts, RLinf copy preparation |
| `patches/` | RLinf flow-noise key patch, applied only to an isolated copy |
| `docs/` | Method contracts and reviews ([index](docs/README.md)) |
| `outputs/` | Coordination, dated plans and selected result bundles (tracked); other run outputs (untracked) |

The package names `fluffy_bassoon` and `lerobot_policy_ctm` are kept from the
source repository so the copied code stays byte-identical; a rename can come
later.

## Environments and tests

The compact learners need only PyTorch and NumPy:

```bash
uv sync
uv run pytest
```

Run tests on allocated Linux resources, not macOS. `tests/test_meta_rl_cuda.py`
runs inside a GPU step. The ManiSkill motor uses the separate RLinf
environment in [docs/maniskill-pi05-icrl.md](docs/maniskill-pi05-icrl.md#environment).

## Artifact retention

For new runs, save checkpoints at required final and evaluation boundaries and
at a fixed update interval declared in the run. Retain only checkpoints that the
paper or live work cites: checkpoints behind reported results with their
sources and initializers, checkpoints awaiting evaluation, checkpoints required
by active or queued work, and the latest complete recovery checkpoint of each
active lineage. Prune other checkpoint payloads after checking live
dependencies. Keep all configurations, logs and evaluation records.

`outputs/`, data, downloaded weights, run worktrees and caches are untracked.
Keep method contracts beside implementations, verified results in
`EVALUATION.md`, and raw evidence in each run's `outputs/<run-id>/`. Never commit
credentials or machine-local authentication configuration.

Planning documents and compact result bundles are tracked exceptions in
`outputs/` (force-add them):

- `outputs/coordination.md` is the single live coordination file: question,
  ground rules, current state, job status and plan. Every agent and server
  updates this file instead of starting another one. Snapshot it to
  `outputs/archive/coordination/coordination-<YYYYMMDD>.md` (links fixed for
  the new location) when a phase closes or it passes about 300 lines, then
  trim it to current work and point it at the snapshot.
- Dated plans, handoffs and working notes live in `outputs/<run-id>/`. Run
  folders are never moved, so links and receipts stay valid. When a run is
  superseded or retired, add one status line at the top of its main document
  naming the replacement.
- Track reports, plans, figures, curve data and audit summaries. Checkpoints,
  raw trajectories, videos and bulk inputs stay untracked; for a file above
  about 5 MB or a copy of another run's file, record its hash and cluster path
  instead.
- Git history keeps everything committed, so tracked scratch files (empty
  locks, partial writes) may be deleted.
