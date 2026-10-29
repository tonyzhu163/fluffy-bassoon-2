# Documentation index

| Guide | Contents |
| --- | --- |
| [In-context RL review](in-context-rl-review.md) | Literature review (RL², AD, PEARL, MELD, AMAGO, TrMRL, Vintix, ECET, VLRLA, survey), pi0.5 language steering, and the proposed CTM in-context RL loop |
| [VLRLA experimental protocol](vlrla-protocol.md) | Exact RoboTwin six/six task split, collection and optimization settings, bridge/simp training, simulation and real-robot evaluation, and reproduction gaps |
| [Frozen-VLA ICRL design axes](frozen-vla-icrl-design.md) | Training method, memory architecture and steering interface; staged DSRL/noise and RL² comparison |
| [CTM in-context RL audit](ctm-icrl-audit.md) | Audit written in fluffy-bassoon: gaps, synchronization/tick risks and the recommended CTM adapter. File paths in it refer to fluffy-bassoon at the time |
| [ManiSkill pi0.5 in-context RL](maniskill-pi05-icrl.md) | Frozen pi0.5 motor, hidden per-trial rotation, outcome cache, training stages, controls, environment, assets and launch commands |
| [Trial environments and PPO](../src/fluffy_bassoon/meta_rl/DESIGN.md) | Trial lifetime and feedback, tasks, recurrent PPO, CTM control, looped Transformer |
| [Continuous Memory Machine](../src/lerobot_policy_ctm/architectures/continuous_memory_machine/DESIGN.md) | The paper-aligned CMM core and its few-shot regression calibration |
