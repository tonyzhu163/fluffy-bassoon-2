# In-context reinforcement learning for frozen VLAs

Can a frozen π0.5 policy improve on a new task using its own earlier attempts? This repository studies that question on **RoboCasa365 and RoboTwin**, with **RLinf** for rollout and online baselines. Current experiments use native simulator rewards and train history models offline. At deployment all weights stay fixed; the learner receives the actions the robot actually executed and the feedback available at that point.

| Layer | Code | Role |
| --- | --- | --- |
| RoboCasa365 | [Launchers and configs](benchmarks/robocasa/) · [adapter](src/fluffy_bassoon/robocasa/) | Atomic kitchen tasks for the current stage; composite tasks reserved for later transfer evaluation. |
| RoboTwin | [Launchers and configs](benchmarks/robotwin/) · [adapter](src/fluffy_bassoon/robotwin/) | Bimanual control with separate source and held-out tasks. Motor release, camera order, action coordinates and controller must match within each comparison. |
| RLinf | [Integration](src/fluffy_bassoon/rlinf/) · [revision-specific patches](patches/) | Native rollout and online PPO/GRPO baselines, action accounting, probability replay and checkpoint recovery. |
| Meta-learning | [Learners](src/fluffy_bassoon/meta_rl/) · [frozen motor interfaces](src/fluffy_bassoon/rl/) | Algorithm Distillation (AD), Decision-Pretrained Transformer (DPT), DIT-style advantage weighting and in-context IQL. [Methods](docs/METHODS.md) identifies which objective each data format supports. |

For baseline scores and weights, see the [RoboTwin phone-stand comparison](docs/FINDINGS.md#sidney-phone-stand-baseline) and the [RoboCasa365 Human300 checkpoint](docs/FINDINGS.md#robocasa365). The [documentation index](docs/README.md) covers the broader project. For a new run, read [AGENTS.md](AGENTS.md) and check [live ownership](outputs/coordination.md). Earlier experiments are reproducible from their recorded Git commits; [provenance](PROVENANCE.md) records the original repository split.

## Getting started

On allocated Linux, `uv sync --locked` installs the shared learner and test dependencies. The simulator/RLinf environments are separate, pinned per benchmark. See [environment setup and tests](docs/METHODS.md#environments-and-tests).

```sh
uv run python -m fluffy_bassoon.meta_rl --help
uv run python -m fluffy_bassoon.meta_rl dpt --help
```

A fit takes separate training and validation archives with hashes and a declared objective. Those commands describe the inputs; they do not launch a run.

## Artifact retention

Keep verified results and links to their evidence in [FINDINGS](docs/FINDINGS.md). Keep live ownership and status in [coordination](outputs/coordination.md). Store dated contracts, configs, logs, evaluation records and compact audit summaries in `outputs/<run-id>/`; never reuse a run ID. Finish or archive work on its original source commit. Historical run scripts use the layout at that commit.

Keep checkpoints supporting reported results and their initializers, checkpoints awaiting evaluation, those needed by active or queued work, and the latest complete resumable checkpoint for each active lineage. Save evaluation/final checkpoints and the recovery interval declared in the run. Before pruning any payload, check dependencies and recovery. Large checkpoints, data, caches and videos stay outside Git; record their hashes and locations. Keep evidence bundles cited by results or needed by another run. Remove superseded plans when neither code, evidence nor live work needs them.

RoboCasa Flow PPO supports opt-in rolling retention through `native_pilot.keep_latest_checkpoints`: keep at least two complete recovery boundaries, plus the initial, evaluation and explicit teacher-export checkpoints. Retiring a superseded boundary verifies hashes and preserves its inventory; it never prunes raw histories. Use `native_pilot.canonical_compression=gzip` for lossless rollout storage. Declare these settings before launch and leave running worktrees unchanged. Share source data across learners and stage bulk data on job-local scratch, with verified cold archives for closed runs.
