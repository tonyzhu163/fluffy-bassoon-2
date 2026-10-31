# Native PPO replication priority

The human prioritized replicating VLRLA's PPO result before further offline
student sweeps. The current experiment must first establish native PPO gain
over its own SFT initializer on matched cases. Frozen-noise PPO is a separate
method. Use the existing RLinf entry/config and the retained dependencies;
preserve the already completed LIBERO outputs and active RoboTwin export.

GRANT.json fixes the initial bounded two-iteration onset/qualification pilot,
all-in ceilings, exact released SFT initializer and explicit four/eight-GPU
execution variants. Eight-GPU capacity is currently fragmented; the four-GPU
variant preserves the global batch by doubling accumulation. This is a stated
hardware deviation, not a smaller scientific rollout/minibatch.

Use the published recipe in docs/vlrla-protocol.md section3 and the existing
RLinf configuration. The implementation owner adds only the necessary explicit
step0 evaluation, clean-boundary worker recovery and conservative action
accounting. Qualify changed entry/config on allocated CPU, then launch within
the fixed six-GPU-hour aggregate ceiling. Verify actual gradients, action-expert
parameter changes, frozen VLM, masks, global batch and complete checkpoints.

The released RLinf SFT/PPO checkpoint pair supplies a useful independent
benchmark reference: official docs report85.94% versus96.09% for pi0.5 on
adjust_bottle. Its exact evaluation protocol must be matched before claiming
those numbers. VLRLA's50.9% versus63.4% headline is for its final bridge;
its table's online-RL comparator is GRPO. Source PPO has an appendix recipe,
not an isolated headline aggregate. Exact author checkpoint/seed matching is
still unresolved, so this pilot is recipe-aligned and cannot be called a full
numerical reproduction. Two iterations do not substitute for the reported25.

The optional released PPO evaluation shares the same hard action/time ceiling
and cannot delay the training entry. All pilot data/weights stay separate from
the existing six-source-task meta-learning bank. After actual timing and matched
success curves, decide the full25-iteration run and its measured resources.
No additional AD100/500 sweep is being prepared or launched.

Primary references:
- https://arxiv.org/html/2610.19517v1#A1.SS2.SSS1
- https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html#visualization-and-results
- https://github.com/RLinf/RLinf/blob/c605b9d55ed93a180119b099e6f836abff293b08/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml
