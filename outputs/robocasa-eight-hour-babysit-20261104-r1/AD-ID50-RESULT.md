# Audited converged AD: three-task ID readout

The validation-selected candidate is update13800 (SHA989e87fe); training stopped at14800 under the original validation plateau. No evaluation outcomes selected or changed the checkpoint.

This is a direct-action baseline using the frozen original VLM features. Its small trained action head supplies physical commands. The later learned-KV steering experiment through the frozen action expert is a separate conditional stage. Deployment uses zero SGD.

At the first query, AD full-mode success was46/50 on Drawer,28/50 on Toaster and29/50 on Coffee. Matched native SFT counts are35/15/21, and source-task fixed130 teacher counts37/23/24. These are descriptive results from one joint three-task fit with different training data/objectives, not a controlled learning-efficiency comparison.

Full history shows no consistent advantage over resetting between episodes in this panel. At five completed support episodes, full-mode counts are44/27/28 versus reset46/29/30. Each point is the actual next query after its own real preceding episodes; cumulative ever-success is a separate outcome.

| Task / memory | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| OpenDrawer / full | 46 | 45 | 46 | 43 | 43 | 44 |
| OpenDrawer / episode | 46 | 46 | 46 | 46 | 46 | 46 |
| CloseToasterOvenDoor / full | 28 | 23 | 24 | 28 | 22 | 27 |
| CloseToasterOvenDoor / episode | 27 | 28 | 28 | 29 | 28 | 29 |
| CoffeeSetupMug / full | 29 | 32 | 31 | 30 | 32 | 28 |
| CoffeeSetupMug / episode | 29 | 27 | 27 | 27 | 30 | 30 |

All entries are successes out of50 paired cases. AD-ID50-RESULT.json contains every per-query paired contrast and marginal interval; these are descriptive without a multiplicity-adjusted adaptation claim.

Independent CPU audit passed all1800episodes,1080000physical controls,21600chunks,5400videos and3245400decoded frames, including all actual commands/draws/feedback and initial native state matches. The scheduler released job-zqllm5eqzgj3 at04:21:44 Shanghai; conservative charge3.7367GPU-hours.

AD OOD50 was submitted only after this audit passed, as job-8htog8yitbyq. Its outcomes remain pending.
