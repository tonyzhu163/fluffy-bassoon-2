# Research coordination and visibility

The human authorized execution of the memory-control recovery, Flora processor
audit and two independent velocity/all-layer-KV PPO arms, then requested
monitoring at every five PPO updates and assessment of Weights & Biases.
The exact owner chats, cursors, budgets and delivered milestones are in
[STATE.json](STATE.json). The coordinator observes existing owners and does not
launch competing jobs or edit their active run worktrees.

## PPO visibility

Each arm retains the fixed 50-update plan, 384 episodes per update and 600-control
TRAIN horizon. Record an initial DEV panel at update 0, then a full checkpoint
and DEV panel at 5, 10, 15, 20, 25, 30, 35, 40, 45 and 50. There are ten
scheduled progress panels per arm. DEV remains separate from the final TEST.

At each completed panel, report first-success and end-success per task and their
equal-weight five-task means against update 0, with case-paired uncertainty when
the actual case/noise pairing supports it. Accompany the curve with TRAIN
success/return, policy/value losses, entropy/std, joint KL, clip fraction,
explained variance, actual optimizer steps, controls, GPU-hours, throughput and
memory. A missing panel or invalid artifact is a failure to diagnose, not a
reason to fabricate a point or change the scientific schedule.

The coordinator heartbeat runs every 20 minutes and reports newly verified
milestones and meaningful failures/results. It stays quiet on unchanged state.
Owner monitors retain responsibility for actual scheduler handles, GPU probes,
first advancement, resumable checkpoints and terminal release. The durable
milestone list prevents repeated notifications.

## Weights & Biases assessment

W&B is useful for comparing these two long-running arms. The existing driver
already writes `metrics/event-*.json` and `PROGRESS.json` after durable complete
boundaries; a single master-side optional exporter can mirror those metrics
without instrumenting every worker. The production owner owns this integration.

Use one stable run per arm and a common comparison group. Define explicit PPO
update and physical-control counters as chart axes, because W&B's default step
counts logging calls. Preserve run identity on resume and check the installed
SDK's actual resume behavior. Durable local metrics and checkpoints remain
authoritative; remote initialization, sync or logging failures must not destroy
or duplicate a completed update. An offline fallback must not become a launch
dependency.

Verify the installed SDK, existing account/project configuration and outbound
access on Baige without revealing credentials. Upload only an explicit scalar
metric and non-secret configuration allowlist. Disable automatic code/git-diff
and console capture; do not upload profiles/native environments, observations,
videos or checkpoints. Account creation and sharing changes are outside this
integration. Online status and usable dashboard URLs must be verified before
claiming the dashboard is connected.

Official documentation consulted November 3, 2026:

- [Custom axes](https://docs.coreweave.com/products/wandb/track/log/customize-logging-axes)
- [Environment variables, offline mode and run identity](https://docs.coreweave.com/products/wandb/track/environment-variables)
- [Experiment creation and metric logging](https://docs.coreweave.com/products/wandb/track/create-an-experiment)

These are the current redirect destinations of the corresponding official
`docs.wandb.ai` pages. No cloud experiment or upload was created by this
assessment.
