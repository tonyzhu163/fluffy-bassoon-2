# Repository guidance

This repository asks whether a CTM can do in-context reinforcement learning.
It was split from `tonyzhu163/fluffy-bassoon` (branch `lycoreco_rl`); see
[PROVENANCE.md](PROVENANCE.md). LycoReco work and runs pinned to fluffy-bassoon
commits stay there.

Keep this file short and durable. Do not append experiment recipes, dated
decisions, run histories, job IDs, availability snapshots, or session notes.
Put method contracts in `docs/` or the method's `DESIGN.md`, verified results in
its `EVALUATION.md`, dated plans and raw evidence in `outputs/<run-id>/`, and
live status in the single `outputs/coordination.md` (archived per the
[retention policy](README.md#artifact-retention)).

## Development and deployment

- Prefer simple solutions and use `uv` for Python environments. Run model
  computation and tests on allocated Linux resources; use macOS for editing
  and static checks.
- On macOS, run `proxy_on` in the same shell before Git operations. Use
  interactive zsh (`zsh -ic`) when needed to load the shell function.
- Keep the main clone on `main`. Use detached worktrees for changes and
  commit-pinned runs.
- Merge verified feature branches into `main` promptly, and rebase onto
  `main` before starting new work; long-lived parallel branches diverge into
  conflicting copies of the same commits. Push `outputs/coordination.md`
  edits straight to `main`, changing only your own row, never only on a
  feature branch.
- For new runs, review and use the latest code on `main` before launch,
  recording its exact commit and relevant validation. Keep active runs
  immutable and record any later migration explicitly.
- Deploy verified commits through Git. If server access fails, transfer a Git
  bundle; use `rsync`/`scp` for artifacts. Verify the commit on each host and
  never modify an active run's worktree. Reconcile dirty or diverged checkouts
  without overwriting changes.
- Never commit credentials, machine-local SSH configuration, or secret-bearing
  logs. Protect shared data and checkpoints needed by active jobs.
- Launch GPU jobs within the user-approved task scope as soon as verified code
  and allocated resources are ready; no repeated launch confirmation is needed.
  Ask before expanding the scientific scope or a materially larger resource
  budget. CPU tests and static checks need no ask.

## Cluster access

| Environment | Entry point | Rules |
| --- | --- | --- |
| Primary school cluster | `ssh thu1-master` | Use the machine-local alias; do not bypass its authentication/routing with a direct IP. |
| Secondary school cluster | `ssh thu2-node01` | Use configured `thu2-*` aliases; off-campus routing uses `ProxyJump mac-mini`. |
| Baige | Authenticated console or configured development SSH endpoint | The user's scheduler containers are exclusive; use their allocation as the isolation boundary. |
| DeltaAI | `yliu21@dtai-login.delta.ncsa.illinois.edu` | Only requested DeltaAI work; account `bhyv-dtai-gh` (project `CIS261376`), batch partition `ghx4`. |

- Both school-cluster aliases reach campus through the `mac-mini` Tailscale
  jump host. If it loses power it stays off the tailnet until someone logs in
  at its console; running cluster jobs are unaffected.
- The school clusters have separate Slurm controllers and homes. Always
  identify the cluster along with a node or job ID.
- Jiaxuan is reserved for Yuxiang Liu (`yuxiangliu`) and work he explicitly
  authorizes. Use the school clusters otherwise.
- Keep SSH routing and authentication local. For DeltaAI, inspect `ssh -G`,
  reuse a live authenticated control connection, and expect password/Duo
  authentication when it expires. Never store authentication material here.
- Before scheduling, run `bash scripts/gpu_availability.sh` for the school
  clusters and check the target's `sinfo`, `squeue`, partitions and GRES.
  Discover current hardware and limits instead of keeping node inventories here.
- On DeltaAI, specify `--account=bhyv-dtai-gh` on every allocation and validate
  ARM-compatible packages and CUDA on allocated Grace Hopper hardware.

## Storage and environments

- Use project-specific shared NVMe for code, environments, caches and outputs,
  rather than NFS home. Check capacity, quota and a write probe before setup.
- Secondary project root: `/cluster/nvme5/zhujiayang/fluffy-bassoon-2`, with
  the canonical clone in `repo/`, pinned runs in `worktrees/<run-id>/` and
  outputs in `outputs/<run-id>/`. Never reuse a run ID. Environments and model
  caches already under `/cluster/nvme5/zhujiayang/fluffy-bassoon/{envs,cache}`
  are reused in place by path; do not move them while fluffy-bassoon runs use
  them.
- Prefer direct downloads. If blocked, use a Git bundle for private code;
  public mirrors must never receive private-repo credentials. Use the user's
  own Hugging Face authentication for gated weights.

## GPU jobs

- On Baige, retain scheduler GPU visibility without extra device namespaces or
  bubblewrap. For evaluation-only work on an unchanged qualified runtime, use a
  lightweight GPU-count/CUDA check and a real simulator episode.
- On school and DeltaAI clusters, run GPU work inside Slurm allocations and
  `srun` steps, never login shells. Set `--ntasks=1` on both `sbatch` and `srun`
  for single-process drivers. Request the partition's actual GRES type and
  enough CPU/RAM; pin nodes only for a hardware or runtime constraint.
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
