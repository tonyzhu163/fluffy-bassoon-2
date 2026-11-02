# First-success sample stopped by an incorrect frame-count assertion

Baige job-g86nibs7mi39 ended in Failed status. The driver recorded exit1 and
520 worker GPU-seconds. Source d30380f7. No successor inference has yet run.
The model produced eight baseline rows in scoring/revision1_1-00.jsonl;
there are no candidate outputs or completed paired result.

The first two short prefixes correctly have24/50 sampled frames, while later
ones cap at64. Our assertion required64 unconditionally and stopped after
writing the first eight rows, before its batch receipt. This is a run-validation
bug, not a Flora inference failure or a rubric result. PARTIAL-VALIDATION.json
checks every retained row against the pinned official sampler and preserves
its exact source. All eight rows qualify; full-score wrong-target112 appears
at control300, but no new-versus-old comparison is available yet.

The proposed r2 keeps scientific inputs/criteria unchanged, validates sampler
counts and actual decoded prefix arrays on CPU, and reuses these eight checked
baseline labels under a new output identity. All original charges remain.
Raw partial files and transfer hashes are retained, and the remote full log
remains under allocation/scorer.log. Production reward stays unqualified.
