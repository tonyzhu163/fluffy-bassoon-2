# DEV selections locked; independent TEST running

2026-11-04 15:01 Shanghai. All80 validation checkpoint panels are complete.
Allfive selection locks passed the source-model, strict-load, ready-receipt,
128-bank-hash and highest-score/earliest-tie checks. Every task has started
its separate TEST panel after its selection lock.

| Task | Baige Pro6K job | Selected update | DEV initial | DEV selected | DEV130 |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenDrawer | `job-cn2if26i9hth` | 30 | 44/64 | 48/64 | 47/64 |
| CloseToasterOvenDoor | `job-zwstryxe9plb` | 1 | 22/64 | 28/64 | 25/64 |
| PickPlaceDrawerToCounter | `job-sfm20c7jc9dp` | 80 | 26/64 | 34/64 | 21/64 |
| CoffeeSetupMug | `job-49hgr77uuw87` | 20 | 19/64 | 28/64 | 26/64 |
| TurnOffStove | `job-mrwxr4gq2p7u` | 30 | 9/64 | 12/64 | 7/64 |

These are DEV selection counts, not independent improvement estimates.
The maximum over16 checkpoints is optimistic. OpenDrawer30 also ties70,
90,100,110, so choosing30 does not establish a unique convergence point.
The selected model hashes, all80 strict-load/source bindings, complete DEV
scores and allfive original lock objects are preserved in
[the TEST-start record](DEV-SELECTIONS-TEST-START-R1.json). The preceding
[selection audit](DEV-SELECTIONS-LOCKED-R1.json) binds the model and READY
hashes and confirms every observed TEST call starts after its lock.

TEST uses128 disjoint cases/task for each of initial0, selected and final130,
with common sampler RNG and counterbalanced policy order within fixed
eight-case banks. At15:01 there are40/240 completed TEST policy-bank receipts
(Open3,Toaster14,Drawer4,Coffee10,Stove9); no partial TEST success statistic is
used for selection. Allfive native jobs continue with no training or recipe
change. Complete DEV plus current TEST progress is5440 episodes.

Conservative all-attempt creation-to-check GPU usage is12.211 of the22-hour
cap, including349GPU-s qualification. The full initial reservation remains
78049GPU-s/21.680GPUh, including allfive allocations; it is not added again
to elapsed usage. PFS has560.74GiB free, above the200GiB floor. No retry or
extension was submitted. These bounds are not a currency bill or a final
scheduler-release audit.

Next: finish the remaining independent TEST banks, verify every raw join and
hash with the Git-pinned CPU closeout, compute paired statistics and the
five-task Holm correction, then verify scheduler release and final cumulative
cost before publishing conclusions. The original update130 results and
failed teacher gates remain unchanged. No student fitting is admitted.
