# Native SAPIEN command replay on two clean phone-stand scenes

Seeds 100000 and 100001 each reached native success on target 128 in both replays, matching the original Sidney rollout. After a fresh seeded reset and the same prefix of 50 joint targets, both robot articulations, phone and stand poses, observed joints and all three RGB views matched bit for bit on both scenes. The maximum difference remained zero at all 129 sampled boundaries per replay. Both wrist cameras reported 0.010 m from the first observation.

This qualifies seed reset plus command replay for these two clean development scenes on native RoboTwin `fa8d97b0`. An alternative can be tested from target 50 by replaying the first 50 targets and verifying the decision state again. Other tasks, randomized scenes and in-place PhysX snapshot restoration require their own measurements. No policy inference was needed to replay the archived commands.

[Measurements](RESULT.json) · [GPU probe](gpu.json) · [Worker clock](exit.txt) · [Scheduler](scheduler.png). Baige Pro6K `job-itm85kscarh2`, source `00e360f5`, four full episodes and 512 delivered targets. Worker time was 221 seconds on one GPU (0.0614 GPU-hours). After the worker exited successfully, the scheduler still held its container. We requested release. Baige reports 8m09s of task time (0.1358 GPU-hours) and an end time of 12:12:34 Shanghai. Charging from submission gives 0.1953 GPU-hours. [Scheduler accounting](scheduler-close.png).

The three [Linux CPU suites](CPU.json) passed all 44 tests on `31616218`, including actual Hydra composition of Sidney PPO and GRPO, the camera hook, and an end-to-end measured-label student gradient.
