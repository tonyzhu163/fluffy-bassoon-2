# Frozen steering distribution diagnostic

Prepared2026-10-31. Source-only final500 AD/DPT weights are fixed. The same
nine LIBERO90 development cases0,10,…80/state0/seed123 receive63 separate
K0 queries: native; original AD/DPT; zero-mean AD/DPT with unchanged learned
std; original-mean AD/DPT with unit std. All comparators share one motor
process per case. No target data enter pretraining or checkpoint selection.

Question: are learned means and/or dispersion involved in the negative
transfer? This intervention is a diagnosis, not a new trained method or a
meta-learning claim. Preserve actual feedback, cache and one original RNG draw.
Allocated tests plus actual loaded-policy qualification must reproduce the
original actions/cache/RNG bitwise before proceeding.

63×410=25,830 nominal actions including settling, hard27,000 cap. One GPU
with3,600 cumulative seconds includes setup/tests/failures/retries/teardown.
Reassign3,600 from6,140 reconciled unused seconds of original32h approval,
leaving2,540. No source collection or old active contract changes.

Run: `libero40-distribution-diagnostic-20261031-r1`. Exact source, inputs,
cases, operations and hashes are in plan.json; target-only reservation is in
target-ledger.json. Tests and model work run only in the actual allocation.
