# Documentation

[Research design and benchmark](PAPER.md) describes how to train a frozen robot policy to use earlier attempts and how to evaluate it on standard LIBERO and the held-out LIBERO-Pro2.0 tasks.

| Document | What it answers |
| --- | --- |
| [Methods and runtime](METHODS.md) | Which learners, action interfaces, data formats and benchmark integrations exist? |
| [Verified findings](FINDINGS.md) | Where has history helped, where has it failed, and what evidence supports each result? |
| [Learned video reward](FLORA-PPO.md) | What did RoboCasa reward calibration establish, and what would physical-robot feedback require? |
| [RoboTwin protocol](robotwin/rl-baselines.md) | Which task split, motor, action coordinates and runtime define the bimanual baselines? |
| [Literature](literature/) | How do the published objectives, steering interfaces and online baselines differ? |
| [Meta-pretraining scale](literature/meta-pretraining-scale.md) and [hidden-variable inference](literature/hidden-latent-meta-learning.md) | How much diversity has been tested, and what would show that practice identifies a property of an unseen robot? |

The current submission is the [SymMeta ICML paper](../papers/icml-xemb-sysid/main.tex) on cross-embodiment command-interface identification, with its [open work](../papers/icml-xemb-sysid/TODO.md) and [campaign evidence](../outputs/xemb-icml-campaign-20261121-r1/evidence/). For the earlier LIBERO decision-learning campaign, consult the [split](../outputs/libero-meta-learning-20261117-r1/SPLITS.json), [registered run plan](../outputs/libero-meta-learning-20261117-r1/PLAN.md) and [execution record](../outputs/libero-meta-learning-20261117-r1/EXECUTION.md). The [coordination board](../outputs/coordination.md) records owners' last reports. Dated run contracts describe their own experiments. See the [archive](../outputs/archive/README.md) for retired proposals and earlier layouts.
