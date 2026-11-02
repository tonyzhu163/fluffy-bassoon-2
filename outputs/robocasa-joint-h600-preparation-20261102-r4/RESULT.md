# Audited600-control calibration and common-before preparation

Baige `job-al2ztuon33dz` succeeded on four GPUs at **2026-11-02 20:22:24 Shanghai**, pinned to source `b7b9c8996c5124ebb56d06a2986824474b4b471c`. Both stages completed: **250 calibration episodes and grades, then540 common-before episodes; zero SGD**. All four scheduler instances and producer clocks are terminal, with zero automatic restarts.

All five prospectively defined600-control gates pass with both native classes, native success between5% and70%, and AUROC at least0.7. No human exception or old750-control gate was used. The original official Flora prompt and weights remain unchanged.

| Task | Native successes | Flora AUROC |
| --- | ---: | ---: |
| OpenDrawer | 31/50 | 0.933786 |
| PickPlaceCounterToCabinet | 28/50 | 0.905844 |
| PickPlaceDrawerToCounter | 14/50 | 0.982143 |
| TurnOffStove | 7/50 | 0.903654 |
| CoffeeSetupMug | 15/50 | 0.847619 |

| Untouched-policy before panel | Episodes | Native successes |
| --- | ---: | ---: |
| DEV |40|19 (47.5%)|
| ID |250|90 (36.0%)|
| OOD |250|126 (50.4%)|

Each ID task has50 cases: OpenDrawer24, CounterToCabinet24, DrawerToCounter21, TurnOffStove4 and CoffeeSetupMug17 successes. Each OOD task has50 cases: CloseFridge28, OpenCabinet35, TurnOnMicrowave8, TurnOnSinkFaucet21 and OpenStandMixerHead34. OOD retains900/1050/450/600/450 controls, receives no Flora scoring, and the policy has no clock input. These are frozen-initializer descriptive panels across different tasks, not evidence of learning or an ID-versus-OOD causal effect.

The allocated Linux CPU audit verified all790 bank-scoped case identities, original motor-noise plans, deterministic zero residual actions,9,930 complete records,496,500 physical policy controls, and2,370 three-view video hashes and frame counts at native20Hz/256px. Every episode reaches its full horizon with an initial frame. All10,720 physical journal intents are resolved. All250 raw grades,20 shard receipts, model/source references and independent AUROC calculations agree. The four engines loaded once each and replayed eight saved warmup episodes each; those32 labels add no independent cases. Each left23,608MiB free against the21,973MiB four-lane guard.

The common initializer checkpoint preserves residual policy, empty optimizer state, RNGs, sampler/motor state, complete environment recovery and four worker request/result identities. Its tensor fingerprint is `641bc65f296cc3f377f3e92a7ea86c6a8f1756017ae7d351a8be09bb633cee24`. Both inactive joint templates are rebound to this qualified source, all five fresh gates, COMMON-FINAL, the initial-policy fingerprint and initial DEV. **Joint SGD remains NOT_ADMITTED** until its separate finite resource/storage/source bindings are supplied.

The scheduler ran66min25sec. This allocation charged16,179GPU-seconds (4.494GPU-hours); preserving r1 conservative864,r2 charged289 and r3 charged2,115 gives **19,447GPU-seconds =5.402GPU-hours**, within the26,400GPU-second approval. The6,953GPU-second balance was not spent. All earlier qualification and experiment spend remains separate and unchanged.

Execution used four workers with four native lanes/GPU, Flora batch capacity8 and utilization0.75. This completion does not qualify eight or sixteen native lanes/GPU, an eight-GPU colocation run, or a learning benefit. Later comparisons must retain the qualified batching or explicitly qualify an execution migration. The earlier r5 one-GPU stress measurement remains the combined-peak VRAM measurement; this run adds successful four-GPU execution and per-GPU warmup headroom, without inventing a full-run VRAM peak series.

Retain the pinned source worktree, common checkpoint/fingerprint, raw worker requests/results, all native videos and monitoring, all scorer outputs/receipts, journals and terminal clocks: the inactive joint templates refer to them.

[Terminal review](TERMINAL-REVIEW.json), [physical/video/checkpoint audit](TERMINAL-AUDIT.json), [grade/source audit](GRADE-SOURCE-AUDIT.json), [source qualification](SOURCE-QUALIFICATION.json), [scheduler closure](SCHEDULER-TERMINAL.json), [binding review](BINDING-REVIEW.json).
