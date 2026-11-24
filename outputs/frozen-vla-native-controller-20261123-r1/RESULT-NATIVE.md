# Current-pose differential IK transferred five of six frozen-policy successes to joint control

The frozen Panda π0.5 policy succeeded on all six layouts under its source Cartesian controller. Replanning on target observations and using bounded differential inverse kinematics at the current joint posture succeeded on **5/6** with the native joint-position controller. Holding the joints succeeded on **0/6**, as did fitting one fixed inverse from native calibration pulses. Each comparison with IK has five paired gains and zero losses. The physical arm, gripper and mesh remained the same Panda. The tested transfer changes the native action interface from six Cartesian increments plus gripper to seven joint increments plus gripper. [Predeclared transfer contract](PLAN.md) · [All native successes and failures](evidence/gpu01/live-summary.json)

| Task and released layout | Source OSC | Joint hold | Current-pose IK | Fixed calibrated inverse |
|---|---|---|---|---|
| Bowl onto plate, 0 | Success (108) | Failure (220) | Success (204) | Failure (220) |
| Bowl onto plate, 2 | Success (113) | Failure (220) | Failure (220) | Failure (220) |
| Bowl onto plate, 3 | Success (108) | Failure (220) | Success (194) | Failure (220) |
| Bowl onto stove, 0 | Success (85) | Failure (300) | Success (168) | Failure (300) |
| Bowl onto stove, 2 | Success (79) | Failure (300) | Success (161) | Failure (300) |
| Bowl onto stove, 3 | Success (89) | Failure (300) | Success (169) | Failure (300) |

Numbers in parentheses are native 20 Hz controls. Success required the native BDDL predicate and reward. Failure means the full task horizon elapsed. The two sampled tasks gave IK success rates of 2/3 and 3/3. Three layouts per task and one policy-noise draw per layout leave uncertainty about other tasks, starts and inference processes. The historical layout-0 Panda source successes took 106 and 87 controls. Their live repetitions needed 108 and 85, preserving both success labels. [Historical source and replay evidence](RECEIPT.md)

## Saturation and reachability

The IK branch issued at least one joint increment at exactly its advertised magnitude limit on **1,059/1,116 controls (94.9%)**. The calibrated inverse reached the limit on **1,420/1,560 (91.0%)**. The five successful IK episodes took 80–96 more controls than their successful source episodes. Saturation and slower target execution occurred on both successful and unsuccessful starts. They are consistent with difficulty reproducing the source servo’s speed. They do not establish why the remaining plate layout failed. The five successes constructively establish reachability under those target conditions. The sixth failure supplies no evidence that the goal is physically impossible. [Per-episode action and reward readback](evidence/gpu01/readback.json)

The fixed inverse previously reduced local one-step errors relative to holding the joints. Its zero native successes show why that proxy alone is insufficient for selecting a controller decoder. These data favor conditioning the inverse on current pose and testing longer-horizon corrections. The geometric and fitted decoders also differ in their use of response measurements, objectives and calibration protocol. This comparison cannot isolate a single architectural cause. [Original motion and pacing measurements](RECEIPT.md)

## Provenance

Baige Distributed Training job job-xnkurg4zw82v ran on a single NVIDIA RPBZZZ6 (RTX PRO 6000D pool) at revision 7518022c88711815c8207b4397e4ce84036c6134. It exited 0 after 411 seconds of worker wall time, or 0.114 GPU-hours. An independent Linux CPU check read all 24 original reward and issued-action tapes from /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-native-controller-20261123-r1/gpu01. First policy-observation, action-proposal and noise hashes matched across all four branches for each start. Native action dimensionality, bounds, gripper preservation, terminal reward and clock were checked at every control. The compact evidence retains each full tape’s SHA-256. [GPU receipt](evidence/gpu01/gpu.json) · [Exit](evidence/gpu01/exit.txt) · [Readback program](readback_native.py)
