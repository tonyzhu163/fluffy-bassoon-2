# Matched Sidney PPO / GRPO bounded comparison

Updated Nov 6 after the two-iteration native PPO pilot completed. Qualification attempts qual01/qual02 and actor-only grad01/opt01 are closed; opt01 explains a zero-length FSDP optimizer representation difference. One actual gradient update with independent trained-state restore and the full twelve-worker/grouped runtime qualifier remain required. Scientific arms are held. The human cumulative RoboTwin cap remains 50 GPU-hours; see GRADIENT-ADMISSION.json for the bounded next step.

Both fresh arms start from original broad Sidney SFT, never a PPO-updated
expert. Use native mixedBF16/fullFP32vision+norms, three cameras,14D absolute
commands, H400/H50/K5/.3, global2048/micro64/accum8/fiveepochs and the same
native1TiB/batch-end memory route with experimental per-close trimmingOFF.
GAE PPO keeps its critic; GRPO uses grouped policy-only loss and no value head.
The current group1 PPO is engineering evidence, not the matched reference.

CASES.json is a subset of the existing pinned1000-seed TRAIN bank. Python
Random0 shuffles that list once, then selects256 distinct scene groups. Every
round contains32 groups with8 contiguous replicas each;4rounds/iteration and
2iterations give2048 episodes per arm. No evaluation or held-out-task seeds
enter this manifest. Both methods must consume the same immutable case table,
record actual requested/effective seeds, public reset identities, conservative
motor-command slots and distinct case coverage. No new simulator qualification is claimed
by static subset preparation. This explicit case table replaces the earlier
proposed reliance on method-dependent native sampler cursors; the new pair is
a declared local protocol, not a numerical VLRLA reproduction.

Before launch, qualify a runtime wrapper that sets each full source reset to
its manifest row, rejects partial/autoresets, checks group identity, and
recovers the next ordinal from a complete iteration boundary. Bind common
policy-sampling and minibatch RNG streams after model initialization so the
optional critic does not accidentally consume paired policy randomness.
Keep native unexposed physics variability separate from these named RNGs;
no full simulator snapshot or exact closed-loop replay is claimed.

numerical_qualification.py is prepared for allocated Linux only. It compares
native GRPO to independent scalar trajectory-return/sample-standard-deviation
references, checks zero-spread groups and absence of critic returns, and checks
that group8 scene layout leaves native GAE values unchanged on identical data.
It does not qualify model replay, critic-free GPU gradients, frozen VLM or
recovery. Those actual-device checks and an episode are still required after
the current PPO reaches a real optimizer/checkpoint boundary. No fresh broad
SFT benchmark panel or full six-task source campaign is added.

Reserve125GiB per scientific arm and a separate40GiB qualification allowance;
reconcile closed qualification artifacts before admitting the pair. Other
owners/floor286GiB includes the pending40GiB packed-readout reservation. Pair
free-space gates are536GiB aftercurrentPPO closes, or661GiB while conservatively
reserving its full125GiB. Count actual remaining growth without double-counting
already consumed bytes. Each4GPU/170min arm has11.4GPUh maximum including launch
margin; pair22.8 plus at most3qualificationGPUh. Reconcile actual completed
current-run timing and charges before any explicit authored-envelope migration;
never borrow another campaign's budget or exceed50cumulativeGPUh without the
human's approval.

Actual qualification evidence: qual02 passed native learner numerical references, 32 groups × 8 exact within-group public reset/exposed-articulation identities and four old-logprob replays, then stopped before SGD on an invalid None-only intermediate frozen-gradient representation check. The current read-only audit records every frozen view at all eight accumulation boundaries; synchronized gradients must be None, all intermediate views finite zero/empty, and frozen parameters excluded from Adam. These gates still require an actual completed update.

Grad01 then stopped before replay because only the native raw Adam digest differed. Opt01 reproduced the mismatch and measured the complete named state/group delta: only zero-length unowned state was dropped; owned state and all other signatures match exactly. New signatures retain the raw digest and an exact named comparison that validates empty dropped state. The next actor-only grad02 initializes fresh original SFT, requires all six legacy initializer signatures to equal qual02 checkpoint0, replays its immutable canonical batch, saves an actual step1 boundary and restores it in an independent process. No old fixture payload is modified. Full qual03 must subsequently qualify simulator groups/raw-reference deduplication, all twelve workers, and next case ordinal1.

The active authored engineering envelope is 23 additional GPU-hours, human cumulative 50 unchanged. GRADIENT-ADMISSION.json reconciles closed actual charges and reserves one GPU-hour for grad02 and three for full qual03. The prospective 46-hour additional science envelope remains inactive. Both science arms await the actual gradient/full-runtime gates and fresh 536 GiB storage/resource admission. Post-model named actor/rollout RNG seed bases are 22000/23000 plus rank; actor.seed1234 controls model initialization.

Control accounting limitation: requested/conservative native14D motor commands bound the escrow. The current fast path has measured_valid_mask=False; actual executed actions, post-success no-ops and internal physics substeps are not measured. The bounded pair may claim common native case/horizon, not measured physical-control matching. Full source/student corpus admission remainsFalse until passive duration/mask/raw action alignment is qualified under a fresh reviewed source without changing policy/loss/reset behavior. See parent train07/CONTROL-ACCOUNTING.md.

Each bounded science arm also performs an independent actor-only readback of its actual final step40 boundary after training, including PPO's critic optimizer group. This adds no SGD, simulator episodes or changes to the saved native contract. It stays within the original170minute/11.4GPU-hour per-arm ceiling; failure of this final recovery keeps job-level qualification false. The already running qual03 checkout is immutable.
