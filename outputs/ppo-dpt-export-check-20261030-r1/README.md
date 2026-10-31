# PPO AD/DPT exporter validation

Implementation tested: `9e2bb54a6bdc6caa6bee1ce757153189eac721d0`.
Tests and inference ran on Baige Linux CPU in the configured development
container `aibox-rffe453e3884-5f96c78b88-rtvhb`, with CUDA hidden and
Torch2.7.1. No simulator, motor inference, new GPU allocation or training
campaign was launched for the saved-data smoke.

## Verified behavior

- 66 relevant tests pass, including the new exporter tests and existing PPO,
  recurrent policy, history-dataset and corpus/training regressions.
- The terminal single-task PPO qualification exports289 decisions from8
  committed PPO trials as paired AD/DPT data. All actual context fields match.
  Repeating the export gives byte-identical AD and DPT NPZ files.
- The terminal shared PPO qualification exports125 decisions across3 actual
  tasks as DPT data. It does not claim40-task exposure. Requesting AD correctly
  rejects this source: each task has only one fixed-behavior trial.
- All83 pinned inputs for the single-task source and33 for the shared source
  remain hash-identical after export. Teacher memory is rebuilt using real
  executed feedback and the frozen teacher's reset convention.
- Source recovery costs are recorded once across paired objectives. The
  original collector journals retain qualification/retry accounting. New
  simulator actions and VLA queries are both zero.

Evidence: [test output](pytest-r3.log), [saved-data script](smoke.py),
[receipt](real-data-receipt.json), [CLI output](real-data-smoke-r3.log).
Datasets remain under this run ID on Baige; source runs remain unchanged.
The first test pass exposed an incorrect test assumption that all actions
after an episode reset match even when actual prior actions differ; the test
was corrected. A saved-data pass then identified older episodes without a
per-row task key. Compatibility uses their exact committed summary task
identity and checks any per-row task keys when present.

This validates the export path, not teacher expertise, robot improvement or
in-context learning. Current teacher policies remain candidates. The existing
SAC all40 assembly and active source runs were not migrated.

Usage: [PPO corpus contract](../../docs/frozen-vla-libero/PPO-CORPUS.md).
