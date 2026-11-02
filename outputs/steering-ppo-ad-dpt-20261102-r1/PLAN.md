# Steering PPO → converged AD/DPT → ID/OOD attempts1–5

User objective: monitor the scale-up experiments; after each completed five-task PPO source, train AD and DPT to validation convergence and evaluate actual attempts1–5 on every ID and OOD task. Also monitor the corresponding Stage1 AD/DPT experiments on both interfaces. This goal remains open until the complete source-specific fits, native evaluations, figures and costs are verified. Earlier single-task action-residual results do not complete this goal.

## Source campaigns and ownership

Keep the Stage1 OpenDrawer pilot and the five-task scale-up as distinct source identities unless a later direct human instruction explicitly consolidates them. Both interfaces are velocity and independent all-layer K+V, with the VLA/VLM/action expert frozen. The old `conditioning_all`32-D value-only handoff is not full K+V and cannot be relabeled as it. No PPO/source job is launched by this monitoring chat; the existing owner chat01a19b74 implements and launches its authorized sources. No duplicate jobs, source edits or unrequested task/architecture expansion.

At initial inspection the scale-up owner is actively implementing/qualifying its driver, not collecting a verified five-task source yet. Its preparation plan names104 balanced80-episode rounds,16 episodes/task/round, 5M training/qualification controls per arm, and50 held-out cases on each of ten tasks. This working-copy observation is not an immutable source binding or runtime-success claim. Read REGISTRY.json and revalidate the live owner/job handle before drawing a new status conclusion.

Five ID tasks: OpenDrawer, PickPlaceCounterToCabinet, PickPlaceDrawerToCounter, TurnOffStove, CoffeeSetupMug. Five OOD tasks: CloseFridge, OpenCabinet, TurnOnMicrowave, TurnOnSinkFaucet, OpenStandMixerHead. Training/ID horizon600, H50/ten flow steps; OOD source plan horizons900/1050/450/600/450. Bind the actual launched source and final case-bank/horizon contract before evaluation. Never infer readiness from this preparation snapshot.

## Actual source data

Each source gets a new immutable binding after authoritative terminal training/checkpoint/evidence verification. Require complete behavior-version coverage and complete actual episode/chunk files, full final PPO policy/normalization, exact frozen base/norm/interface mapping, and case/draw provenance. Failed or partial episodes stay charged but cannot silently become source rows. Do not use the completed Flow SDE action-expert run as velocity/KV steering data: it has neither steering-code labels nor the same frozen expert.

The source adapter uses2064 pre-steering public features and actual pre-tanh steering coordinates:32 velocity,1152 independent K/V coefficients (18 layers×2 targets×32). Native iid Gaussian draws remain explicit exogenous provenance, not trainable steering targets. Older joint-noise-action1632-D policies are a different contract and are rejected.

History training uses every chunk of each actual complete physical episode, including the physical post-success suffix. Rewards stay exactly at their recorded first-success chunks; the history done flag is the recorded native physical_done. The optimizer's shorter first-success logical PPO prefix is separately checked against those same physical records and preserved. No reward is moved to the end, no suffix is invented, and no source physical interaction is dropped from accounting. This full-physical history view must be used consistently during the attempt evaluator.

Group one task/collection slot chronologically across source behavior versions; never concatenate different tasks into one trial. Select12.5% held-out source slots with fixed PCG64 seed90211 before constructing windows/labels, preserving each task in both partitions. Source PPO saw these validation states; this is a student-imitation split, not an independent teacher-training split. Source cases across versions may differ; do not describe them as same-scene retries.

AD target=actual steering u. DPT target=frozen final source PPO mean u at the exact same saved feature, normalization and interface. All non-target arrays must be bitwise identical. Teacher state/config and each source checkpoint are hash-bound. PREPARE.py writes all four datasets and only then a completed assembly receipt.

## Independent convergence fits

Primary fits are AD/full and DPT/full separately for each interface and source campaign. Retain existing Stage1 fits owned elsewhere when the exact source contract matches; monitor rather than duplicate them. No unrequested static-model fit or backbone sweep is introduced. Use the existing plain causal decoder: width128,2 layers,4 heads,context300,logical batch8,AdamW3e-4,seed0. Train on random windows of1–5 complete physical episodes; do not truncate them at64 tokens. Capacity must cover five complete attempts on the largest bound evaluation horizon (currently105 tokens), and the first real batches must demonstrate the declared window sampler.

Validate on16 fixed held-out batches every100 updates. Minimum1000 updates, then10 checks without a material decrease greater than.001×max(1,abs(reference NLL)); select the lowest observed validation NLL. FIT.py uses --until-converged, not a fixed update count. The12000-second invocation and scheduler time limits are operational checkpoints, not convergence or an aggregate scientific resource cap. Full latest model/Adam/sampler/Python/NumPy/Torch/CUDA RNG/plateau payloads must be verified before resuming the identical source/data/optimizer contract under a fresh allocation attempt. Do not refit an already-converged model.

A fit publishes as soon as its own plateau is verified. Its own native readout can begin immediately; no all-objective or all-interface barrier. Require actual allocation probes and advancing updates, not queue state alone.

## Per-task native curves

Repeat the user's actual-attempt protocol:50 prospective cases per task, attempts1–5, frozen weights, physical reset to the same case between attempts, controller memory retained within the case and cleared between cases/methods. Preserve the source's frozen base/norm/map and source-declared actual first-success feedback. Record first-success and success-at-end separately. Keep every physical chunk through the declared horizon and install no future reward before its actual observation. No extra scored query episodes enter the primary curve.

AD and DPT share the exact canonical case bank and per-attempt native Gaussian streams; different attempts use the prior protocol's distinct fixed streams. Exact native state/XML/cameras/public state must match before action; original pixels retained with only a qualified render-roundoff bound. Source PPO's before/after panels remain separate comparisons if their streams differ. Do not infer a causal memory benefit from an attempt trend without a separately authorized matched reset/noise control. No unapproved tripling of the evaluation cohort is implicit in this goal.

The five-task primary cohort is2500 actual episodes/model/interface (50cases×10tasks×5attempts),10000 across two models/two interfaces; report both task-level curves and equally weighted five-task ID/OOD macro averages. The Stage1 pilot cohort remains separately bound to its actual task scope. Preserve all completed prefixes and partial-call costs across recoveries; never replay discarded query feedback. Each model/phase must close its own allocation and summarize on allocated Linux CPU independently.

## Completion evidence

- Both exact interface sources and any distinct Stage1 source identities accounted for, with live→terminal scheduler evidence, immutable final checkpoints and complete actual steering histories.
- Each requested AD/DPT fit reaches the declared validation plateau; selected/latest hashes, full recovery state and source/data/sampler contract verified.
- Every task's50-case native attempt1..5 cohort verified from actual simulator truth, full case/draw/source/map hashes and memory-isolation/frozen-weight evidence.
- Independent model results published, then combined ID/OOD per-task/macro PNG/SVG/CSV with95% Wilson intervals, paired changes/tests and source/context/seed/render limits.
- Full source/fit/readout/failure/recovery costs and terminal resources; source checkpoints/worktrees retained; only verified outcomes added to docs/FINDINGS.md and only this chat's coordination row changed.
- Keep the goal and recurring monitor active until all requirements are proven. A producer observation timeout does not mean its job stopped; revalidate the same handle before retrying or escalating.
