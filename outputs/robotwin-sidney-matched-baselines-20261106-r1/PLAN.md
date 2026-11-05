# Matched Sidney PPO / GRPO bounded comparison

Prepared2026-11-06 while the independent engineering PPO run is active.
No GPU job has launched from this directory. The current training checkout and
contract remain immutable. Launch gates and proposed budgets are in the
parent pilot's MATCHED-PAIR-CONTRACT.json; engineering-envelope migration is
not active, and the human50GPU-hour cumulative RoboTwin cap still applies.

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
record actual requested/effective seeds, public reset identities, physical
controls and distinct case coverage. No new simulator qualification is claimed
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
margin; pair22.8 plus at most2qualificationGPUh. Reconcile actual completed
current-run timing and charges before any explicit authored-envelope migration;
never borrow another campaign's budget or exceed50cumulativeGPUh without the
human's approval.

Prepared wrapper source now includes case_schedule.py, paired_recipe.py and
paired_entry.py. Only syntax/static source review has run on macOS. No model,
learner numerics, group reset, GPU gradients or recovery qualification has run
for this wrapper. Its launcher and actual-device qualification remain gated;
these files do not constitute admission or a completed comparison.
