# Finish the saved KV comparison

The user explicitly raised the old campaign cap from 150 to 170 GPU-hours on
November 12: "Raise the old cap to 170; finish the comparison". Evaluate the
saved DPT, DIT and IC-IQL models against the completed AD reference. This is
9,000 new native episodes, not a refit. Budget is at most 22.5 GPU-hours for
three single-GPU jobs of at most 7.5 hours each, including startup.

The recorded historical total was 140.416111 GPU-hours before five November 9
pilot allocations. Reserving their entire configured limits conservatively
as 3 GPU-hours gives 143.416111 before this readout and 165.916111 including
the three new reservations, below the approved 170. Exact historical charges
will replace the conservative bound after scheduler reconciliation. All new
work, including the three teacher reservations and two failed starts, reserves
47.406667 of the separately authorized 50 GPU-hours.

Preserve the original native runtime, selected checkpoints, canonical scenes,
seeds, fixed B8 waves, horizons, frozen expert, zero deployment SGD, six real
attempts and full/reset conditions. OOD remains Fridge/Microwave, held out of
the three-task meta-training corpus. It does not imply absence from VLA SFT.
Use the exact original independent rollout auditor and reducer. Report all
five task curves, paired full/reset differences and uncertainty; keep next
attempt success distinct from cumulative ever-success.

The current main code provides a new operational launcher. READOUT.py differs
from the original only in its CPU watcher heartbeat directory. PACKED-WAVES.py
changes only its owned run directory. PACK-CLOSED.py changes that same directory
and the storage floor to 200 GiB plus 3 GiB packing space. The scientific
implementation is pinned to 87984718, because these saved models belong to
that qualified runtime. This migration does not edit the closed historical run.

GPU jobs write a bounded hot bank buffer on shared PFS. A Linux CPU watcher
independently audits each closed bank, copies its tar to private BOS, checks
the complete archive hash, restores it and compares every member hash before
releasing redundant hot raw data. Result/audit metadata stay at their original
paths. A failed audit or stale watcher stops new waves; incomplete data and all
checkpoints remain protected. Shared input storage is writable because stock
RoboCasa writes transient XML beside assets. New PPO teachers wait until the
comparison has enough storage headroom; the two workflows do not share data.

Validate deployment on Linux, verify actual scheduler hardware and GPU probe,
then observe a completed native bank before treating a job as progressing.
Check each active job at most 20 minutes after the previous check. No automatic
retries or output identity reuse. Record source, job IDs, progress, usage and
next action in EXECUTION.json and the existing coordination row.
