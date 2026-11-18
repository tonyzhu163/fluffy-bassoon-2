# Few-shot adaptation of frozen robot policies

Can a robot use its own previous attempt to improve its next one on a held-out manipulation task? This project trains a small history-conditioned controller offline. The robot brings a frozen vision-language-action model to the task and adapts by remembering what it did, what moved and whether it succeeded. We use the standard **LIBERO four-suite score** for comparison with published VLAs and **LIBERO-Pro2.0** for transfer. The Pro2.0 design measures independent-episode robustness and the benefit of memory in paired retries on held-out tasks. [Research design](docs/PAPER.md)

| Part of the project | Entry point |
| --- | --- |
| Benchmark and split | [Published LIBERO-Pro2.0 evaluator](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/docs/remote-evaluation-guide.md) · [Exact fitting, development and test IDs](outputs/libero-meta-learning-20261117-r1/SPLITS.json) |
| Decision pretraining | [Support histories, query targets and motor interfaces](docs/METHODS.md) · [Offline learner implementations](src/fluffy_bassoon/meta_rl/) |
| Evidence | [RoboCasa365 and RoboTwin baselines, controlled history experiments and held-out comparisons](docs/FINDINGS.md) |
| Simulation and online RL | [RoboCasa365](benchmarks/robocasa/) · [RoboTwin](benchmarks/robotwin/) · [RLinf integration](src/fluffy_bassoon/rlinf/) |

Start with the [documentation index](docs/README.md). For execution, use the [repository guidance](AGENTS.md) and [coordination board](outputs/coordination.md). The [provenance record](PROVENANCE.md) documents the original repository split.

## Getting started

On allocated Linux, `uv sync --locked` installs the shared learner and test dependencies. The simulator/RLinf environments are separate, pinned per benchmark. See [environment setup and tests](docs/METHODS.md#environments-and-tests).

```sh
uv run python -m fluffy_bassoon.meta_rl --help
uv run python -m fluffy_bassoon.meta_rl dpt --help
```

These commands inspect the offline archive trainer. A fit takes separate training and validation archives with hashes and a declared objective. The LIBERO benchmark uses its own simulator and policy environment. [LIBERO campaign record](outputs/libero-meta-learning-20261117-r1/EXECUTION.md)

## Artifact retention

Keep verified results and links to their evidence in [FINDINGS](docs/FINDINGS.md). Keep live ownership and status in [coordination](outputs/coordination.md). Store dated contracts, configs, logs, evaluation records and compact audit summaries in `outputs/<run-id>/`; never reuse a run ID. Finish or archive work on its original source commit. Historical run scripts use the layout at that commit.

Keep checkpoints supporting reported results and their initializers, checkpoints awaiting evaluation, those needed by active or queued work, and the latest complete resumable checkpoint for each active lineage. Save evaluation/final checkpoints and the recovery interval declared in the run. Before pruning any payload, check dependencies and recovery. Large checkpoints, data, caches and videos stay outside Git; record their hashes and locations. Keep evidence bundles cited by results or needed by another run. Retire superseded plans when code, evidence and live work no longer require their paths. The [documentation archive](outputs/archive/README.md) explains how to recover earlier plans.

RoboCasa Flow PPO supports opt-in rolling retention through `native_pilot.keep_latest_checkpoints`: keep at least two complete recovery boundaries, plus the initial, evaluation and explicit teacher-export checkpoints. Retiring a superseded boundary verifies hashes and preserves its inventory; it never prunes raw histories. Use `native_pilot.canonical_compression=gzip` for lossless rollout storage. Declare these settings before launch and leave running worktrees unchanged. Share source data across learners and stage bulk data on job-local scratch, with verified cold archives for closed runs.
