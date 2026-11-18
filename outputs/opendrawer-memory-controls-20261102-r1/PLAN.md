# Matched empty-memory and 64-token controls

Human request, 2026-11-02: isolate memory from motor resampling, quantify
native noise before more source histories, fix inference context length,
park Flora optimization and reduce run bookkeeping.

Use the exact frozen AD/full and DPT/full checkpoints and the held-out
50 OpenDrawer / 50 CloseFridge cases from the completed actual-attempt run.
Keep original p1..p5 motor streams, physical reset seeds, horizons (750/900),
logical case ranks, motor batching, pixels and final simulator success.
Neither fitting nor PPO nor new source histories is in this run.

Two controls per model, each reporting all 500 endpoints:

1. **Episode reset:** clear all state and previous feedback before each of
   five attempts; retain normal within-episode context. Reuse verified p1
   episodes, which already start empty, and collect p2..p5 (400/model).
2. **Full history, raw-token window64:** retain cross-attempt feedback but
   rebuild from the last64 raw tokens whenever necessary. This avoids older
   history surviving in higher-layer cached representations. Reuse full
   history episodes only before the first affected query (ID attempts1..4,
   OOD attempts1..3); collect ID5 and OOD4..5 (150/model). Check replayed
   state against original at the reuse boundary. Total fresh:1100 episodes,
   915000 policy controls, plus constructor/reset controls charged as observed.

New controls use latest reviewed main with unchanged checkpoint weights.
Validate core controller reset/window behavior on allocated Linux, plus a
real matched simulator episode and actual allocated-GPU count/CUDA check.
Keep old worktrees immutable. Bind checkpoints, case bank and motor seeds
directly from the completed original bindings; record new source pin at launch.
No extra receipt framework or bookkeeping tests. Stop on a failed real check
and diagnose before assigning a fresh output identity.

Report per-attempt success separately from ever-success. Compare full versus
reset on each paired case/stream, and capped versus full on affected attempts;
also report difference in attempt5-minus1 between full and reset. Use exact
paired tests and case-cluster intervals across five correlated attempts,
with multiplicity and exploratory status disclosed. Existing50-case curves
remain development evidence. Confirmatory comparisons require150–200 fresh
cases or multiple independent draws per case/attempt, chosen before seeing
results; those are not silently added to this minimal matched diagnosis.

First summarize all four completed native8 audits (80 selected anchor-failure
cases,640 episodes). Save each case's k/8, Wilson interval,0..8 histogram and
pairwise disagreement. Conditional failure selection and different horizons
must be disclosed. Empirical best-of8 recovery is an oracle headroom diagnostic,
not a universal ICRL ceiling.

Flora optimization remains parked until simulator-reward improvement passes
and the observed false-full-score failure problem is corrected and qualified
on held-out failures and successes. Do not launch more history collection.

Request four2-GPU workers (8 GPUs total),64 CPUs/256GiB RAM/32GiB SHM per worker, for at most180minutes (86400 GPU-seconds). The driver has a175-minute operational deadline and preserves partial records on failure. Source/checkpoint binding and core checks run before GPU allocation. Resource binding and scheduler ID will be added only after actual allocation.
No automatic retries; monitor launched work every20minutes.

Deployment preflight: direct HTTPS Git fetch succeeded on Baige; the detached
run checkout is pinned to `2250ac96b087fde2bc8164aff8b91d45df005e82`.
The original1000 episode/checkpoint/seed/truth bindings passed, and the15 core
controller tests passed on allocated Linux CPUs in2.83s, CUDA hidden
(`core-tests-2250ac96.log` in the shared output directory). Shared storage has
637GiB available and the binding write succeeded. No source files were copied.

Baige accepted `job-kw6ub7t73wri` at19:48 Shanghai on2026-11-02, queue
`aihcq-rh0wwk9muei4`: four2-GPU RPBZZZ6 instances with64 CPUs/256GiB RAM
and32GiB SHM each,180-minute cap, automatic fault recovery off. It is queued;
GPU probes and first matched simulator episodes are not yet verified. A
20-minute chat heartbeat monitors capacity, actual progress and terminal
artifacts, stays quiet on unchanged status, and stops after completion or
terminal failure. The partial web-terminal bundle transfer was abandoned;
use direct Git fetch for source synchronization.

First allocated check, Nov2 21:57 Shanghai: the job started at21:39. All four
2-GPU probes passed with eight distinct allocated GPU UUIDs, and all eight
workers were advancing.119 saved fresh episodes passed source/checkpoint,
record/truth hash, simulator pairing and reset/window-limit checks. The live
shared output `MONITOR.json` contains this check; no worker terminal failures
were present. Continue20-minute checks and retain the original source pin.

Terminal failure, Nov2 22:14:02: `job-kw6ub7t73wri-worker-1` exited1.
Worker05 reached its first ID case and the current environment guard rejected
the historical750-control OpenDrawer horizon because current prospective
`TASKS` uses600. The preflight checked controller behavior but missed this
cohort/registry mismatch.267 completed AD/OOD episodes remain (184 reset,
83 window64); no ID or DPT control episode completed. All containers stopped.
Charge at most36minutes ×8 GPUs =17280 GPU-seconds, including interrupted
episodes. The distinct r2 continuation binds750/900 locally in the evaluation
process, checks both constructors on CPU before allocation, verifies/reuses
the267 completed rows, and has a140-minute8-GPU cap. Combined maximum84480
GPU-seconds remains below the original86400. Preserve this failed worktree
and its original binding; no in-place retry or historical horizon change.
