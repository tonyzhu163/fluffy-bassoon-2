# Joint three-task baseline qualification

2026-11-04 Asia/Shanghai. Code pin: `cc469f320cb3f0750e9e2f7b0e348b3272ffc4fd`,
Git-bundle deployed to the clean Baige worktree
`/mowenjie/fluffy-bassoon-2/sources/three-task-joint-cc469f32`.

- The synthetic pooled-export check passed for all eight objective/split datasets.
  It verifies exact field arrays, within-stream chronology, distinct namespaced
  task/stream identities, source hashes and disjoint training/validation streams.
  It uses four-chunk fixtures with the real 112/16 stream counts per task;
  full source extraction and GPU execution remain unqualified.
- 47 existing Linux CPU tests passed: 15 flow-history tests in 8.79 seconds,
  plus 32 history/value tests in 37.15 seconds. These cover causal histories,
  action/label semantics, value loss references and next-Adam checkpoint recovery.
- Python AST, shell syntax and JSON checks pass. No new permanent tests were added.
- All four fitting objectives require a pooled three-task assembly. Per-task
  jobs prepare data only. Fit slices retain the same operational bound across
  exact resumes, preserving the existing source/data/optimization contract.

Evidence: [pooled export](CPU-JOINT-QUALIFICATION.json),
[flow tests](CPU-CORE-TESTS.log), [history/value recovery tests](CPU-RESUME-TESTS.log),
[static checks](STATIC-PREFLIGHT.json).

**Human launch hold was released.** The first two scheduler submissions failed
before data preparation; see EXECUTION.json. The 96 GPU-hour ceiling
is approved; allocation duration is an operational checkpoint,
while the stopping rule is each method's declared validation plateau.
