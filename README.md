# Learning to Steer: In-Context Reinforcement Learning for Frozen VLAs

Can a history-conditioned steering policy adapt a frozen vision-language-action
model to held-out robotic tasks with high sample efficiency and strong final
performance? The paper stays about in-context RL. **The current first step is
PPO with a learned video reward** ([Flora-Rewarder](docs/FLORA-PPO.md)),
tested on LIBERO and reported on RoboCasa365.

**Start here:** [docs/README.md](docs/README.md) indexes the five documents:
[paper plan](docs/PAPER.md), [findings to date](docs/FINDINGS.md),
[Flora PPO](docs/FLORA-PPO.md) and [shared methods and code](docs/METHODS.md).
Agents and collaborators also read [AGENTS.md](AGENTS.md) (clusters, storage,
GPU rules) and the live [coordination file](outputs/coordination.md).

Other earlier work, including CTM pilots, ManiSkill, RoboMimic and most
of the LIBERO source40 pipeline,
was removed on 2026-11-01 because it produced no usable evidence; git history
keeps it. Literature reviews are in [docs/literature/](docs/literature/).
RoboTwin docs are in [docs/robotwin/](docs/robotwin/), and all RoboTwin runs,
plus dead runs that code still loads, are in [`outputs/archive/`](outputs/archive/).

The repository was split from `tonyzhu163/fluffy-bassoon` (branch
`lycoreco_rl`) on 2026-10-28; [PROVENANCE.md](PROVENANCE.md) lists every copied
file and its source hash. The package names `fluffy_bassoon` and
`lerobot_policy_ctm` are kept so the copied code stays byte-identical.

## Environments and tests

```bash
uv sync
uv run pytest
```

Run tests on allocated Linux resources, not macOS. `tests/test_meta_rl_cuda.py`
runs inside a GPU step. Robot runtimes are separate environments; see
[docs/METHODS.md](docs/METHODS.md#environments-and-tests).

## Artifact retention

For new runs, save checkpoints at required final and evaluation boundaries and
at a fixed update interval declared in the run. Retain only checkpoints that the
paper or live work cites: checkpoints behind reported results with their
sources and initializers, checkpoints awaiting evaluation, checkpoints required
by active or queued work, and the latest complete recovery checkpoint of each
active lineage. Prune other checkpoint payloads after checking live
dependencies. Keep all configurations, logs and evaluation records.

`outputs/`, data, downloaded weights, run worktrees and caches are untracked.
Keep the five core documents in `docs/` (plus `docs/literature/` and `docs/robotwin/`) ([index](docs/README.md)): per-run
contracts in each run's `PLAN.md`, verified results in `docs/FINDINGS.md`, and
raw evidence in `outputs/<run-id>/`. Never commit
credentials or machine-local authentication configuration.

Planning documents and compact result bundles are tracked exceptions in
`outputs/` (force-add them):

- `outputs/coordination.md` is the single live coordination file: question,
  ground rules, current state, job status and plan. Every agent and server
  updates this file instead of starting another one. Snapshot it to
  `outputs/archive/coordination/coordination-<YYYYMMDD>.md` (links fixed for
  the new location) when a phase closes or it passes about 300 lines, then
  trim it to current work and point it at the snapshot.
- Dated plans, handoffs and working notes live in `outputs/<run-id>/`. When
  a run finishes, keep the folder only if FINDINGS cites it or live work loads
  it. Otherwise delete it, or move it unchanged to `outputs/archive/<run-id>/`
  if code still loads it. Git history keeps deleted runs; never reuse a run ID.
- Track reports, plans, figures, curve data and audit summaries. Checkpoints,
  raw trajectories, videos and bulk inputs stay untracked; for a file above
  about 5 MB or a copy of another run's file, record its hash and cluster path
  instead.
- Git history keeps everything committed, so tracked scratch files (empty
  locks, partial writes) may be deleted.
