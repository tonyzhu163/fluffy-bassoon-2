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
