# Documentation

Five core documents, read in this order, plus two reference folders:

| Doc | What it holds |
| --- | --- |
| [PAPER.md](PAPER.md) | The research question, paper outline, and the agreed design and claim boundaries |
| [FINDINGS.md](FINDINGS.md) | Every result worth keeping, with evidence links, and the lessons that shape the next step |
| [FLORA-PPO.md](FLORA-PPO.md) | Learned video reward (Flora-Rewarder) PPO, benchmark roles and reward qualification; RoboTwin also has a simulator-reward baseline arm |
| [METHODS.md](METHODS.md) | Shared motors, steering interface, DSRL recipe, LIBERO protocol, code layout and tests |
| this README | Map and conventions |
| [literature/](literature/) | Reviews of published work: [in-context RL literature review](literature/in-context-rl-review.md), [offline ICRL for a frozen VLA](literature/frozen-vla-offline-icrl-review.md), [VLRLA protocol](literature/vlrla-protocol.md), [DSRL recipe audit](literature/dsrl-recipe-audit.md), [frozen-VLA steering scoping review](literature/vla-steering-review.md), [RoboCasa PPO recipes and performance](literature/robocasa-ppo-review.md), [RoboTwin PPO evidence and restoration gap](literature/robotwin-ppo-review.md), [pretraining scale](literature/pretraining-scale.md), [scale references](literature/scale-references.md) |
| [robotwin/](robotwin/) | Active [RLinf / VLRLA baseline comparison](robotwin/rl-baselines.md), [setup](robotwin/vlrla.md), and historical [setup evaluation](robotwin/vlrla/EVALUATION.md) / [steering evaluation](robotwin/coefficient-steering/EVALUATION.md). Earlier run folders stay in `outputs/archive/`; new work uses fresh run IDs |

Live jobs and owners are in [`outputs/coordination.md`](../outputs/coordination.md).
Cluster, storage and GPU rules are in [AGENTS.md](../AGENTS.md).

## What is in `outputs/`

- **Top level:** live runs, the ML10 and DSRL result bundles, and the earlier
  run folders that live runs load directly.
- **[`outputs/archive/`](../outputs/archive/):** coordination snapshots, plus
  dead-experiment folders that existing code still loads by path. Use them only
  as inputs; don't extend them.
- **Everything else was deleted on 2026-11-01**, along with the old method
  docs, because those experiments did not produce usable evidence. Old links in
  remaining files point to GitHub permalinks at the pre-cleanup commit. To
  restore a file, run `git show <that commit>:<path>`.

## Conventions

- Keep the core doc count at five; `literature/` and `robotwin/` are reference folders. Put a new method contract in the run's
  `PLAN.md`, and add a durable choice to PAPER.md or METHODS.md.
- Add a result to FINDINGS.md only once it is verified and well controlled,
  with a link to its run evidence. Keep raw evidence and cost receipts in the
  run folder.
- When a run finishes: if its result is worth keeping, leave it and link it
  from FINDINGS.md. Otherwise delete it, or move it to
  `outputs/archive/<run-id>/` if code still loads it. Never reuse a run ID.
