# Documentation

[Research design and benchmark](PAPER.md) describes how to train a frozen robot policy to use earlier attempts and how to evaluate it on standard LIBERO and the held-out LIBERO-Pro2.0 tasks.

| Document | What it answers |
| --- | --- |
| [Methods and runtime](METHODS.md) | Which learners, action interfaces, data formats and benchmark integrations exist? |
| [Verified findings](FINDINGS.md) | Where has history helped, where has it failed, and what evidence supports each result? |
| [Learned video reward](FLORA-PPO.md) | What did RoboCasa reward calibration establish, and what would physical-robot feedback require? |
| [RoboTwin protocol](robotwin/rl-baselines.md) | Which task split, motor, action coordinates and runtime define the bimanual baselines? |
| [Literature](literature/) | How do the published objectives, steering interfaces and online baselines differ? |

For the current LIBERO campaign, consult the [split](../outputs/libero-meta-learning-20261117-r1/SPLITS.json), [registered run plan](../outputs/libero-meta-learning-20261117-r1/PLAN.md) and [execution record](../outputs/libero-meta-learning-20261117-r1/EXECUTION.md). The [coordination board](../outputs/coordination.md) records owners' last reports. Dated run contracts describe their own experiments. See the [archive](../outputs/archive/README.md) for retired proposals and earlier layouts.
