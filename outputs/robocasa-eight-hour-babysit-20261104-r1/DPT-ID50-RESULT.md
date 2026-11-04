# Audited converged DPT: three-task ID readout

The validation-selected candidate is update16200 (SHA bab8e0f9), also the validation-plateau stopping update. The original fit and all four readout replicas released successfully. Evaluation outcomes did not select the checkpoint.

This direct-action baseline uses frozen original VLM features and a learned physical-action head, with zero deployment SGD. Learned KV steering through a frozen action expert remains a separate conditional stage.

Full history shows no consistent benefit over resetting between episodes. With five completed support episodes the full-mode counts are44/27/27 versus reset45/29/28 for Drawer/Toaster/Coffee. Each point measures success in the actual next query; cumulative ever-success is recorded separately.

| Task / memory | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| OpenDrawer / full | 45 | 44 | 44 | 45 | 45 | 44 |
| OpenDrawer / episode | 46 | 46 | 45 | 45 | 45 | 45 |
| CloseToasterOvenDoor / full | 29 | 28 | 27 | 27 | 28 | 27 |
| CloseToasterOvenDoor / episode | 28 | 29 | 29 | 30 | 29 | 29 |
| CoffeeSetupMug / full | 27 | 31 | 25 | 26 | 29 | 27 |
| CoffeeSetupMug / episode | 27 | 27 | 28 | 27 | 28 | 28 |

All entries are successes out of50 paired cases. Per-query full-minus-reset contrasts, exact McNemar p-values and marginal paired bootstrap intervals are in DPT-ID50-RESULT.json. These remain descriptive, without a multiplicity-adjusted adaptation claim.

Independent CPU audit passed all1800episodes,1080000physical controls,21600chunks,5400videos and3245400decoded frames, including actual commands/draws/feedback, native initial-state matching and memory continuity. Audit receipt: DPT-ID50-AUDIT.json. Scheduler job-h62ufmu061kl ended05:07:23 Shanghai; conservative allocation charge3.6633GPU-hours.

The same frozen candidate completed the contracted CloseFridge H900 and TurnOnMicrowave H450,50cases/task,6actualepisodes,full/reset panel. Independent audit passed; see DPT-OOD50-RESULT.md for the weak transfer result.
