# VLRLA protocol and initial frozen-VLA ICRL decisions

Recorded 2026-10-29, Asia/Shanghai. Paper version: arXiv:2610.19517v1.

- [Published protocol](../../docs/vlrla-protocol.md): tasks, all three data
  routes, model variants, reported optimization/evaluation settings and gaps.
- [Our design](../../docs/frozen-vla-icrl-design.md): three axes; initial
  GTrXL/PPO noise steering; balanced source tasks; K4 training / K8 evaluation,
  with episode indices 1/2/4/8. The user accepted this initial trial-length
  choice after distinguishing it from the evaluation budget.
- [First-results brief](FIRST-RESULTS.md): existing measured results and their
  limits; no new model result is claimed by this documentation work.
- [Public demonstration inventory](robotwin-datasets.json): all twelve task
  identifiers, paper split, immutable HF revision, archive bytes and hashes.
- [Source manifest](source-manifest.json): downloaded paper/guide identities,
  visual PDF checks and bounded release search.
- [Starting evidence receipts](starting-evidence.json): source result files
  and hashes used for the brief.

The official dataset revision inspected was
`TianxingChen/RoboTwin2.0@3dc3b798668feb99ac61cc9086d84cbcc3d79186`.
The twelve `demo_clean.zip` files total **3,998,857,282 bytes** (3.724 GiB);
six source-task files total 1.833 GiB. This is an inventory, not a claim that
the trajectory archives have been downloaded. Source metadata, the paper PDF
and HTML are cached locally under `sources/`, outside the tracked bundle.

No official VLRLA learning-history dataset, code or checkpoint was identified
in the paper's links or the bounded GitHub/HF search. The single GitHub name
match was `1ranGuan/vlrlab-web`, not identified as an official release; HF
model and dataset queries returned no VLRLA matches. Negative search results
are time-bounded evidence, not proof that no artifact exists.

Raw paper downloads, extracted text, figures and rendered pages remain
untracked. Compact factual records and hashes are the durable artifacts.
No GPU allocation or model computation was launched for this source audit.
