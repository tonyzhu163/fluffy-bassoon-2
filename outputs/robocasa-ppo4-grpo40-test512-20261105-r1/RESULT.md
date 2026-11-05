# OpenDrawer: four-epoch PPO40 versus GRPO40

Four-epoch PPO succeeded on 358/512 OpenDrawer cases (69.92%) versus 328/512 for GRPO (64.06%). The paired difference was +5.86 percentage points (95% paired bootstrap interval +1.37 to +10.35); PPO won 87 discordant cases and GRPO won 57, exact two-sided McNemar p = 0.0153705. The primary paired test favors four-epoch PPO for these two checkpoints.

| Fixed checkpoint | First success / 512 | Successful at the final control / 512 |
| --- | ---: | ---: |
| Original SFT reference | 314 (61.33%) | 304 |
| Four-epoch PPO40 (ablation E) | 358 (69.92%) | 351 |
| Stage 1 GRPO40 | 328 (64.06%) | 319 |

Final-control success and comparisons to SFT are descriptive. First success within 600 controls is primary. Each learned checkpoint used 40 rollouts and 3,072,000 fresh training controls. PPO saw 128 distinct starts per rollout and took 960 optimizer steps; GRPO saw 16 starts with eight attempts each and took 240 optimizer steps. This compares the actual action-matched recipes, with one training seed per method. Neither checkpoint was retrained or chosen on this panel.

Both policies and SFT ran on fresh seeds 85,411,000–85,411,511, in the same eight-case groups, with common sampler RNG and counterbalanced policy order. The allocated-Linux audit verified 1,544 complete episodes including the eight-case SFT repeat, all physical/public initial-state matches and 2,316 sampler chunk calls. All 926,400 charged controls were reconciled, including 0 reset controls. Both worker likelihood checks had zero initial chunk log-ratio error. All 667 original GRPO actor tensors were byte-identical in its strict-load inference adapter; its eight unused value tensors do not train or change the actor.

SFT first-success outcomes repeated on 8/8 repeated cases; 0/12 decoded action-batch hashes repeated exactly. The result establishes matched starts and sampler draws, not bitwise reproducibility of every subsequent simulator trajectory.

The [earlier epoch ablation](../robocasa-opendrawer-ppo-ablation-3m-20261104-r1/RESULT.md) supports four epochs over one for OpenDrawer. This separate panel compares four-epoch PPO with the existing GRPO recipe. The [broader source campaign](../robocasa-eight-hour-babysit-20261104-r1/SOURCE5-BUDGET.json) uses four epochs for Counter-to-Cabinet and Faucet. Both broader runs saved checkpoint 10/40 and stopped during the scheduled development evaluation: the simulator XML fingerprint differed on some cases although their public initial state matched. The campaign owner is investigating ([Counter](../robocasa-eight-hour-babysit-20261104-r1/SOURCE5-COUNTER-FAILURE-01.json), [Faucet](../robocasa-eight-hour-babysit-20261104-r1/SOURCE5-FAUCET-FAILURE-01.json)). Replication across training seeds, tasks and longer training durations remains open.

Baige job `job-hxgnpj0csx9s` succeeded and both GPUs were released. One attempt used 3.498 of 7.5 GPU-hours on a conservative creation-to-release basis, with zero new training. [Prospective plan](PLAN.md) · [checkpoint and case manifest](MANIFEST.json) · [paired counts and intervals](ENDPOINT-RESULT.json) · [raw per-case outcomes](TEST512-CASES.jsonl) · [independent audit](ENDPOINT-AUDIT.json) · [cost and release](BUDGET-CLOSEOUT.json).
