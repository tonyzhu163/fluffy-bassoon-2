# Matched corrected-PPO three-to-six task comparison

The user approved 230 additional GPU-hours, accepted the balanced six-task pilot, and then requested a consistent PPO setup. Recollect Drawer, Toaster and Coffee with the corrected critic. Their historical 130-update buggy-critic data remains a reference and is excluded from both new corpora. Preserve the original November15 Shanghai readout target; do not restart the deadline.

## Sources and matching

The six tasks are OpenDrawer, CloseToasterOvenDoor, CoffeeSetupMug, PickPlaceCounterToStove, TurnOffStove and TurnOnSinkFaucet. This gives two tasks in each of three broad manipulation families; it is still a narrow atomic-task pilot. CloseFridge and TurnOnMicrowave remain held out from both new corpora.

Use the same Human300 SFT initialization, frozen VLM, corrected RLinf commit8f9bfa76, action/observation transforms, native first-success reward, 600-control horizon, 50-control chunk, ten denoising steps, noise0.3, ignore_last=true, four PPO epochs, 128 episodes/update, logical minibatch256, learning rates5e-6/1e-4, gamma0.99 and lambda0.95. Match actor initialization seed1234 and use registered disjoint task scene seeds. Each task contributes its first30 updates: 3,840 training episodes and2,304,000 controls. Use the fixed step30 teacher for labels; no endpoint selection by evaluation success.

Start Drawer, Toaster, Coffee and Stove fresh. Extend the qualified Counter-to-Stove step10 checkpoint by20 updates. Reuse Faucet updates1–30 only if its resolved scientific settings, complete chronology and step30 checkpoint verify; exclude its later ten updates and step40 labels. Operational archive paths and checkpoint cadence may differ; verify that these do not change learner settings or training RNG. Any failed reuse check requires a recorded revised collection reservation within the total cap before launch.

The corrected three-task corpus must be the exact hash-bound Drawer/Toaster/Coffee subset of the six-task corpus. All four methods receive matching splits and objective-appropriate labels from the same source data. Do not reuse normalization or Q/V estimators fitted to the old buggy-critic sources.

## Fitting and readout

Collect276,480 frozen-feature/query-label records across the six sources. DPT uses actual step30 teacher actions on saved query observations, never behavior actions as substitutes. Keep complete source streams together for training/validation.

Train eight fresh KV fits: AD, DPT, DIT and IC-IQL on each corpus, each at10,000 updates with seed0, the same architecture, logical batch8 and fixed final-update selection. Sample tasks uniformly within each corpus. The comparison measures adding task data at fixed fitting compute; it does not isolate diversity from corpus size. One fit per cell does not establish seed robustness.

Prioritize five shared evaluation tasks: three ID plus Fridge/Microwave OOD. Use40 matched cases/task, six real attempts, full versus episode-cleared history, and all eight fits:19,200 episodes. This replaces the initial eight-task50-case plan to fund corrected recollection. Report the resulting uncertainty; omit the separate added-task panel from the primary bounded readout. Report first-attempt success, sixth-attempt success and paired history effects separately. Better initial competence alone is not evidence of adaptation.

## Bounded execution

| Stage | Additional GPU-hour reservation |
| --- | ---: |
| Four fresh30-update sources (20 each) and Counter continuation (16) | 96 |
| All-six frozen features and DPT labels | 18 |
| Eight policy fits and corpus-specific value estimators | 62 |
| Shared ID/OOD evaluations and archival | 48 |
| Qualification and recovery | 6 |
| Total hard cap | 230 |

Fresh sources request8 GPUs,64 CPUs,1024GiB RAM and32GiB shared memory for at most150 minutes; continuation requests120 minutes. Preserve full optimizer/sampler/RNG checkpoints and verified private BOS archives. Fresh jobs request a checkpoint at120 minutes, stop physics at130 minutes and reserve the remaining scheduler window for archival. Current estimates derive from the prior ten-update pilots (about53–57 minutes with archival), Faucet's30-update continuation (about115 minutes before archival),52.7 GPU-hours for eight fits and21.17 GPU-hours per9,000 evaluation episodes. They are not guaranteed runtimes. Reallocate unused stage reservations only within230, based on execution cost rather than observed scientific outcomes. The earlier50/170 caps stay closed.

Review the latest main commit and validate resolved configurations on allocated Linux before launching. Recheck queue capacity, limits and actual device type; probe every actual GPU allocation. Use local scratch for bulk raw output, with private BOS verification, because shared PFS has limited headroom. Verify advancing updates after launch and check active jobs every20 minutes.

[Contract](CONTRACT.json) · [Approval](APPROVAL.json) · [Coverage](TASK-COVERAGE.md) · [Fresh source](teacher_fresh.py) · [Counter continuation](teacher_resume.py).
