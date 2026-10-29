# First actual RoboTwin source collection

Baige `job-tgjz72b93hes`, source `daa3bb0324065952d81a7af36ed9ebe7c839db7c`,
ended cleanly with exit0 at22:58:56 Shanghai,2026-10-29. Scheduler duration
16m51s; driver status is **wall_budget_stop**, not full training completion.

Collected **32 episodes,11,475 executed primitive actions and374 chunk
decisions**. Source SAC:12episodes/4,274actions/139replay rows. Persistent
GTrXL/PPO:12episodes/4,273actions/3pending trials. Reset GTrXL/PPO:
8episodes/2,928actions/2pending trials. Only three of six source tasks have
completed trials so far; no held-out evaluation has run. **All optimizer
counters are zero**: SAC has not completed500chunk warmup and PPO has not
filled its256complete-K4-trial batch. This establishes working collection,
not learned steering or across-episode adaptation.

The allocation probe,40 focused Linux tests and real native/cached sampler
parity passed (max error0.0). Source recurrent old-policy replay also matched
exactly. All12 paired initial-observation groups match across methods. The
frozen motor's before/after fingerprint is unchanged. Full coordinator and
three learner snapshots include five pending PPO trials,139SAC transitions,
all optimizer/sampler/RNG payloads and the next method index2. Their four ZIP
CRCs passed; SHA256-verified copies were mirrored locally. Optimizer states
are correctly empty because no gradient step occurred. See
[saved-state inspection](RECOVERY-INSPECTION.json) and [audit](AUDIT.json).

Measured source collection is about12.7executed actions/second on one serial
simulator. Extrapolating this short three-task sample to all three10.24M caps
implies roughly672GPU-hours of collection alone, excluding optimization and
evaluation. This is a rough runtime estimate, not an approved continuation
budget. The original VLRLA-scale recipe therefore needs more efficient
collection before a larger run.

Next gate: optional final-observation-only camera capture retains the native
full-observation action-count loop and its render/RNG side effects. It remains
disabled by default and unqualified until repeated reference episodes and
candidate episodes agree on pixels/features, decoded actions, outcomes and
exact executed counts. The8-minute gate fits within the original45GPU-minute
envelope using measured terminal durations; it earns no source-data credit.
No material compute extension or altered training batch is authorized here.
