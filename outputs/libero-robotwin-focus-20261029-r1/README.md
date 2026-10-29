# Current focus: learned steering on LIBERO and RoboTwin

2026-10-29 20:21 Asia/Shanghai. The user paused the Meta-World fixed-bank
screen and directed work to LIBERO and RoboTwin. Existing campaign allocations
retain their owners; this chat does not submit duplicates or alter their
active sources and runtimes.

## LIBERO

`Plan VLA in-context learning` owns the source40 campaign. Read-only Baige
inspection confirms the bounded `libero40-pi05-source-smoke-20261029-s00-train-r2`
qualification exited 0: 60 episodes, 15 cases, 606 query chunks, 5,796 primitive
actions plus 610 settling actions, and 2,220 source-SAC updates. The final
publication records paired AD/DPT datasets. Compact observed counters and
reported recovery/export hashes are in [libero-observed.json](libero-observed.json);
these hashes are reported by the collector, not independently revalidated here.

This is actual learned noise steering after the warmup, not selection among
16 fixed tensors. It qualifies the runtime and data path only. Success was
59/60 on one source task, so these observations do not establish a useful
recovery curriculum, adaptation benefit, full40 coverage, or 40→90/Plus/Pro
transfer. The owner is preparing full coverage and reviewing scale. Follow
the latest full40/full90/Plus/Pro protocol; older semantic exclusions are
archival rather than launch gates.

## RoboTwin

`Clarify DSLR online RL post-training` owns the active matched comparison in
`/Users/mowenjie/.codex/worktrees/vlrla-protocol/fluffy-bassoon-2`. Its first
job `job-c6u3tk8cf2y7`, source `a7aacde`, stopped before tests/training because
the GPU visibility variable was absent. The owner is preparing a separately
identified retry with the usual exclusive-container visibility fallback.

Static inspection confirms the implementation chooses a continuous noise
action from current features at each chunk, passes it into the frozen flow
sampler, and updates noise SAC using observed transitions. The comparison
also has recurrent GTrXL/PPO and an episode-reset control. Source training
uses the six VLRLA source tasks; target evaluation uses six tasks excluded
from controller training. The selected motor has prior exposure to all50
RoboTwin tasks, so the transfer claim concerns the controller.

The prepared retry keeps the original scientific contract: equal10.24M
source primitive-action ceilings, K4 source trials and K8 evaluation at
1/2/4/8 attempts. Its43-minute allocation is a bounded continuation segment,
not evidence that those ceilings have been completed. Full recovery is
required at its boundary. Next evidence must be actual simulator progress,
finite advancing optimizer updates and retained recoveries, followed by
independently measured benefit over native/static steering. No such benefit
is established yet.

These two campaigns test learned noise steering. Meta-World's failed fixed
bank is not an admission gate for them. No additional allocation was launched
by this focus switch.
