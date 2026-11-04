# Evaluation stopped at the human's request

2026-11-04. The human asked what the remaining evaluation answers and then
instructed: **stop the test cases**. The five exact owned campaign jobs were
checked in the existing Baige console and a batch stop was submitted.
Four jobs reached ManualTermination at15:41; Toaster had already succeeded
at15:39. Allfive allocations are terminal and no campaign GPU remains active.
The four remaining CPU stagers were ownership-checked and terminated; Toaster's
stager had completed naturally. The recurring monitor was deleted through the
automation API. No other owner's job was changed.

| Task | Job | Scheduler terminal status/time, Shanghai | Completed TEST policy banks | Saved cases per initial/selected/final |
| --- | --- | --- | ---: | --- |
| OpenDrawer | `job-cn2if26i9hth` | ManualTermination15:41:23 | 36/48 | 96/96/96 |
| CloseToasterOvenDoor | `job-zwstryxe9plb` | Succeeded15:39:01 | 48/48 | 128/128/128 |
| PickPlaceDrawerToCounter | `job-sfm20c7jc9dp` | ManualTermination15:41:21 | 36/48 | 96/96/96 |
| CoffeeSetupMug | `job-49hgr77uuw87` | ManualTermination15:41:23 | 45/48 | 120/120/120 |
| TurnOffStove | `job-mrwxr4gq2p7u` | ManualTermination15:41:23 | 44/48 | 120/120/112 |

All80 DEV panels and their model-bound selections remain complete. TEST
stopped with209/240 complete policy-bank receipts,1672 completed TEST
episodes, and6792 completed DEV+TEST episodes overall. Interrupted raw files
and unresolved action intents remain preserved; no unfinished bank is counted
as a complete episode panel. OpenDrawer, Drawer-to-counter and Coffee each
retain one unresolved journal call, charged conservatively. Stove has no
unresolved call but its last bank lacks a complete panel receipt. Four native
exit clocks were not produced during scheduler termination; they are not
treated as exit0 or scientific completion.

The [stop closeout](USER-STOP-CLOSEOUT-R1.json) records scheduler timestamps,
every completed bank/episode SHA256, case identities,600-control episode
checks, preserved journals and staging-process release. This is a stop-boundary
audit, not the planned full raw-tensor/statistical closeout. Toaster's native
RESULT and successful clock are retained, but the complete five-task TEST128
analysis was not run. Private model copies for interrupted tasks are retained.
All source checkpoints and corpora remain unchanged.

Final conservative creation-to-terminal allocation usage is **55746 GPU-s =
15.485 GPU-hours**, including349GPU-s qualification and every attempt. This
is below the22GPU-hour cap; no running/queued campaign reservation remains.
It includes scheduler startup/teardown and is not a monetary bill.

## What the evaluation answers

DEV shows the checkpoint trajectory and selects a candidate policy. The
independent TEST cases were intended to establish whether that selected
checkpoint actually beats the initial policy on fresh cases, rather than
benefiting from selecting the maximum of16 noisy validation scores. They
would not, by themselves, identify which PPO setting caused instability.

The existing curve and [recipe diagnosis](RECIPE-DIAGNOSIS.md) are sufficient
to motivate controlled critic/update-schedule corrections. They do not prove
those corrections work, or confirm teacher improvement across allfive tasks.
No full five-task paired TEST/Holm claim, teacher promotion or student fitting
is made from this interrupted sweep. Original update130 results and failed
teacher gates remain visible and unchanged. No further TEST cases or retries
are authorized by this stopped campaign.
