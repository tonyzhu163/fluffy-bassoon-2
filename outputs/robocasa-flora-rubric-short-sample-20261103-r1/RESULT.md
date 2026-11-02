# Short direct task formulations: terminal tradeoff remains

Baige job-0z3jwm37ngly succeeded/released00:17:00 Shanghai Nov3, source34d3d647.
All24 grades on two diagnosed terminal failures and four success controls
completed. Baseline hard grades reproduce6/6; all copied file hashes match
Linux.460 worker /472 running /482 all-in upper GPU-seconds, below900.
No new robot actions, motor calls or PPO updates.

| Criterion | Full-score failures /2 | Accepted success controls /4 |
| --- | ---: | ---: |
| Verbatim1.1 |2|3|
|1.6 short uppermost-target instruction |1|0|
|1.7 short final-state instruction |2|1|
|1.8 short two-condition verifier instruction |2|0|

Only1.6 corrects wrong-drawer112 (5→1), while still accepting undone opening64.
It removes all three previously accepted terminal controls128/264/144.1.7
retains only144;1.8 retains none. No candidate meets the prespecified rule of
correcting a diagnosed failure without losing an accepted control. No prompt
is promoted or claimed as a reward repair.

Some controls end with poorly visible target drawers, while a matching
first-success arm would have stopped earlier. The justified next check is
causal-prefix scoring at native first success on these same six retained
trajectories, keeping the corrected target instruction as a candidate and
using only saved actions on CPU to recover missing first-success clocks.
This changes the scientific readout from terminal completion to first success;
it must not retroactively rename terminal false positives as first-success
errors. See ../robocasa-flora-first-success-rubric-20261103-r1/PLAN.md.
