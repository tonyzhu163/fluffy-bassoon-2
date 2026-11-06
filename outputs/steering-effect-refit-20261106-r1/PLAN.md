# Measure steering, then continue a corrected KV learner

The user authorized the next step on 2026-11-06. First measure all eight
preserved candidates with the fixed reset comparison. This tests the
negligible-steering hypothesis; similar robot success counts do not establish it.

The [contract](CONTRACT.json) fixes the exact source/native identities, the
original 128 validation queries, and three 1% screening thresholds before
measuring candidates. These are practical screening choices, not an adaptation
claim. Publish every model's native/full/reset losses and steering/history
magnitudes regardless of whether it passes. The diagnostic uses saved source
observations, zero simulator controls and zero optimizer updates.

Then continue the primary KV-AD learner from a copied complete terminal
recovery. Replay its original scheduled validation losses under the corrected
relative rule, explicitly recording the changed stopping contract. Preserve
weights, Adam, sampler and global/flow RNG; never edit the old finalized fits.
Keep the source, objective, architecture, batch, optimizer and validation query
plan fixed. A new candidate must reach its corrected validation stop and pass
the prospectively fixed gate before any simulator readout. Bind that result to
the exact candidate, new fit contract and validation/native identities at
dispatch; a helper function alone is insufficient.

The parent campaign closed at 140.271111 GPU-hours. Its 150 GPU-hour ceiling
still controls this work: at most 9.728889 new GPU-hours including all attempts,
setup and queue intervals. Start with at most one GPU-hour for the eight-model
diagnostic. Reserve later stages only after measured timing and the remaining
budget admit them. No budget is borrowed from RoboTwin or SOURCE70.

Use the latest verified main commit, a new immutable scheduler source checkout,
and a real job-step GPU probe. Shared PFS must retain its 220 GiB free floor
plus this run's remaining growth and any concurrent reservations. Reuse the
protected complete PFS source cache. Monitor active allocations at most every
20 minutes and through terminal; preserve full checkpoints at operational stops.
