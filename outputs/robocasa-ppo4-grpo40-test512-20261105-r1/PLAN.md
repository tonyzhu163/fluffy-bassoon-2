# Four-epoch PPO40 versus GRPO40

Prospective November 5, 2026. The user authorized reusing the completed
four-epoch PPO checkpoint for a paired comparison with GRPO, then adopting
four PPO epochs for the broader source campaign.

Evaluate fixed final40 PPO arm E and fixed final40 Stage 1 GRPO, plus their
original SFT reference, on 512 fresh common OpenDrawer cases. Both learned
actors trained on 3,072,000 controls. PPO used four optimizer epochs and 960
optimizer steps; GRPO used one epoch and 240 steps. This compares these
action-matched recipes and one training seed each. It does not isolate the
algorithm from optimizer reuse, establish equivalence, or rank general PPO
against general GRPO. No checkpoint selection or training is included.

Cases 85,411,000–85,411,511, sampler seed 95,411,000 + bank index, horizon600,
fixed B8 membership. These cases are disjoint from prior TRAIN, DEV and TEST
banks. Cycle all six policy permutations across banks and repeat SFT on bank0.
Use shared native physical/public starts and identical actual sampler RNG per
chunk. Audit strict checkpoint loads, all episode receipts and resolved action
journals. Reuse exact preserved GRPO actor tensors in its previously qualified
inference adapter; the eight unused value tensors do not affect its actor.

Primary endpoint is PPO4 minus GRPO first-success probability, exact two-sided
paired McNemar at .05, paired case-bootstrap 95% interval. Declare a directional
winner only if p<.05 and the difference has that sign. SFT and terminal-success
contrasts are descriptive. A null result does not establish equivalence.
The independent Linux CPU audit recounts raw episodes and verifies pairing.

Use a wrapper committed on reviewed latest main, retaining qualified immutable
evaluation dependencies 81caf870, engine02ce5940 and vendor8f9bfa76. The dependency
pins preserve the runtime already used for both checkpoints; no active worktree
or saved training contract is changed. Reverify exact model/config/fixture and
boundary hashes on allocated Linux CPUs, plus 667 GRPO actor tensor identities.

Two isolated one-GPU replicas, each16CPU/64GiB RAM/8GiB SHM, 225minute walltime,
automatic retry off. Run the actual allocated GPU probe and SFT likelihood
parity before the first real bank. Verify advancing complete banks and monitor
every20minutes. Nominal926,400 evaluation controls including the8-case repeat;
resets, failures and release remain charged. Require200GiB shared free storage
throughout and verify a write probe before submission.

Separate all-attempt ceiling7.5GPU-hours. This falls within the user's requested
evaluation and the repository's launch allowance, including the earlier
below8GPU-hour rule. It does not borrow the closed92GPU-hour ablation budget or
the separate approved70GPU-hour source budget. Stop at the hard ceiling;
diagnose any failed attempt before retrying under a new output identity.

The broader source campaign is already owned by chat01a1a717 under the approved
[70GPU-hour proposal](../robocasa-eight-hour-babysit-20261104-r1/SOURCE5-PROPOSAL.md)
and [authorization](../robocasa-eight-hour-babysit-20261104-r1/SOURCE5-BUDGET.json).
It trains Counter-to-Cabinet and Faucet for40updates with four epochs and tests
those plus fixed Coffee130. This comparison creates no duplicate source jobs
and does not expand that campaign to new seeds or130update runs automatically.
