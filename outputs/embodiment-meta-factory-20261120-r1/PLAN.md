# Decisions before scaling embodiment meta-learning

The question is whether a public interaction history can select a useful correction to one frozen motor while preserving performance when the correction should be zero. All physical-arm references below refer to simulated robot models. The factory starts at revision `29f1db5581f1a674002cebff2ef723342e8a6604`. The independent source motor collection and SFT remain in the [dual-arm run](../libero-paligemma-dualarm-20261120-r2/PLAN.md).

| Workstream | Owner (local Codex CLI worker) | Question | Evidence |
| --- | --- | --- | --- |
| Motor and correction | motor | Can the same frozen motor realize native task success and can the allowed correction rescue failures? | [Motor](MOTOR.md) |
| Public observations | information | Which body properties predict behavior from a free-space or contact history after visual identity and scene are controlled? | [Information](INFORMATION.md) |
| Identity action | identity | Can a learner preserve exact raw commands when correction is unnecessary, and recognize when to intervene? | [Identity](IDENTITY.md) |
| Meta-training scale | scale | What changes with the number of independent dynamics settings and robot kinematics at fixed transitions? | [Scale](SCALE.md) |
| Critical review | review | What would make an apparent transfer or nominal noninferiority claim spurious? | [Review](REVIEW.md) |

Each worker has a 0 GPU-hour evidence-review allocation and a four-minute target. Save prior results with denominators, the cheapest discriminating test, alternatives and the observation that changes the research decision. The initial factory deliverable is a decision memo, with measured results distinguished from prospective experiments. Large meta-training is downstream of these tests. Raw and adapted controllers use the same PaliGemma SFT file. UR5e remains reserved for final target evaluation, IIWA is held in reserve, and real hardware is deferred.
