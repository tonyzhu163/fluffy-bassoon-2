# Repository guidance

This repository studies in-context reinforcement learning for frozen VLAs; the
current first step is learned-reward (Flora-Rewarder) PPO. Start with
[docs/README.md](docs/README.md). It was split from `tonyzhu163/fluffy-bassoon`
(branch `lycoreco_rl`); see [PROVENANCE.md](PROVENANCE.md). LycoReco work and runs pinned to fluffy-bassoon
commits stay there.

Keep this file short and durable. Do not append experiment recipes, dated
decisions, run histories, job IDs, availability snapshots, or session notes.
Keep `docs/` to its five core documents plus `literature/` and `robotwin/`. Put per-run contracts, dated plans and raw
evidence in `outputs/<run-id>/`, verified results in `docs/FINDINGS.md`, and
live status in the single `outputs/coordination.md`. Delete or archive finished
runs per the [retention policy](README.md#artifact-retention).

## Development and deployment

- Prefer simple solutions and use `uv` for Python environments. Run model
  computation and tests on allocated Linux resources; use macOS for editing
  and static checks.
- Keep only tests that protect core behaviour (learners, models, motors,
  environments, checkpoints, GPU probing) or numerical qualification (values
  matching a reference). Do not add tests for one run's plans, admission,
  receipts, provenance, launch scripts or campaign bookkeeping; check those
  in the run itself and record the evidence in `outputs/<run-id>/`.
- On macOS, run `proxy_on` (or `proxy_on_clash` if that proxy is down) in
  the same shell before Git operations. Use interactive zsh (`zsh -ic`) when
  needed to load the shell functions.
- Keep the main clone on `main` and commit routine changes (fixes, docs,
  configs, small additions) directly to `main`. Use detached worktrees for
  commit-pinned runs.
- Create a feature branch only for a major feature change. Merge it into
  `main` as soon as its tests pass, then delete the branch locally and on
  `origin`. Rebase onto `main` before starting new work; long-lived parallel
  branches diverge into conflicting copies of the same commits. Push `outputs/coordination.md` edits straight to `main`,
  changing only your own row, never only on a feature branch.
- For new runs, review and use the latest code on `main` before launch,
  recording its exact commit and relevant validation. Keep active runs
  immutable and record any later migration explicitly.
- Deploy verified commits through Git. If server access fails, transfer a Git
  bundle; use `rsync`/`scp` for artifacts. Verify the commit on each host and
  never modify an active run's worktree. Reconcile dirty or diverged checkouts
  without overwriting changes.
- Never commit credentials, machine-local SSH configuration, or secret-bearing
  logs. Protect shared data and checkpoints needed by active jobs.
- Launch accelerator jobs within the user-approved task scope as soon as verified code
  and allocated resources are ready; no repeated launch confirmation is needed.
  Do not ask for approval for compute allocations or incremental budgets up to
  50 GPU-hours within the requested task; proceed and record cumulative usage.
  Ask before unapproved budgets above 50 GPU-hours, scientific-scope
  expansion, or exceeding an explicit user hard cap. CPU tests and static
  checks need no ask.

## Environment access

Baige is the default environment for computation, job launches, availability
checks and job-status questions unless the user names another environment.

| Environment | Entry point | Rules |
| --- | --- | --- |
| Baige | Authenticated console or configured development SSH endpoint | The user's scheduler containers are exclusive; use their allocation as the isolation boundary. |
| Ali PPU | Authenticated Alibaba Cloud PAI console or configured DSW development endpoint | Use allocated PPU resources; preserve vendor framework and accelerator libraries, and qualify workloads on the actual device. |

- Keep SSH routing and authentication local. Never store authentication
  material here.
- Before scheduling, check the target's queue capacity, resource limits and
  actual accelerator type. Discover current hardware and limits instead of
  keeping node inventories here.

## Storage and environments

- Use project-specific shared NVMe for code, environments, caches and outputs,
  rather than NFS home. Check capacity, quota and a write probe before setup.
- Never reuse a run ID or move shared environments and model caches while
  active jobs depend on them.
- Prefer direct downloads. If blocked, use a Git bundle for private code;
  public mirrors must never receive private-repo credentials. Use the user's
  own Hugging Face authentication for gated weights.

## Accelerator jobs

- On Baige, retain scheduler GPU visibility without extra device namespaces or
  bubblewrap. For evaluation-only work on an unchanged qualified runtime, use a
  lightweight GPU-count/CUDA check and a real simulator episode.
- On Ali PPU, validate framework/device support and core numerical behavior
  inside the allocated instance before model work; CPU fallback does not
  qualify accelerator support.
- Scheduler availability, GPU health and allocation isolation are separate
  checks. Run `scripts/[train]gpu_probe.py` inside the actual job step with
  `--expected-count` and `--output`. `IDLE` or readable `nvidia-smi` is
  insufficient. Bind inference and rendering to allocated GPUs.
- Use job-local `NCCL_P2P_DISABLE=1` for RTX 6000D distributed work.
- Measure batch capacity and update time. Tune execution batches without
  changing the saved global batch, loss weighting or optimizer schedule.
- After launch, verify advancing updates or completed artifacts, then monitor
  every 20 minutes. Report cluster, job ID, progress, last check and next action
  in task updates. Diagnose failures before retrying with a new output identity.
- Before stopping or moving work, verify ownership and the exact job/step.
  Preserve a complete resumable checkpoint, including optimizer, sampler and
  RNG state. Never terminate another user's work or alter an active run's
  contract.
