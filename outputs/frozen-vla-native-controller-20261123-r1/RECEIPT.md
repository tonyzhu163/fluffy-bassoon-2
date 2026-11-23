# Native joint-controller transfer

**Use bounded differential IK for the first closed-loop comparison. Prioritize actuator timing and grasp feedback over a larger learned inverse.** The frozen OpenPI π0.5 Panda tapes reproduced both native OSC_POSE successes on Linux, in 106 controls for bowl onto plate and 87 for bowl onto stove. The target was robosuite’s factory JOINT_POSITION on the same seven-joint Panda. It received seven legal joint increments plus the original gripper command. Source and target shared the released BASE state and physical chain. [Source tapes](evidence/source) · [Configuration and sensor assumptions](PLAN.md)

| Mean one-control error against achieved OSC motion | Bowl onto plate, mm / degrees | Bowl onto stove, mm / degrees |
| --- | ---: | ---: |
| Hold joints, issue gripper | 3.38 / 0.121 | 3.01 / 0.104 |
| Encoder and model Jacobian, bounded IK | 1.66 / 0.106 | 1.57 / 0.072 |
| Fixed inverse from separate native pulses | 1.50 / 0.262 | 1.70 / 0.230 |
| Query-local response oracle | 0.98 / 0.142 | 1.22 / 0.100 |

IK reduced translation error at all eight sampled postures on each tape. Paired reductions spanned 0.48–2.79 mm on plate and 0.08–2.12 mm on stove. Both calibration response matrices had rank six. The geometric Jacobian agreed with centered FK differences within 4.2 × 10⁻¹¹ per entry. These postures come from two trajectories. [Plate controls and measurements](evidence/libero_spatial.json) · [Stove controls and measurements](evidence/libero_goal.json) · [Pulse calibration](evidence/calibration.json)

None of the three joint decoders reached success before either recorded tape ended. At least one joint command hit its bound on 92% of plate and 84% of stove IK controls. A privileged tracker given the successful source joints reduced TCP error at first gripper closure from 164 to 2 mm on plate and from 154 to 9 mm on stove by timing each segment for the stock servo. Neither schedule earned reward by the native horizons of 220 and 300 controls. The rate-limited plate schedule required 14.71 s for an 11 s episode. The stove schedule fit its 15 s horizon but still missed. This motivates a controller that advances the grasp sequence using measured motion and finger state. Its task benefit remains open. [Trajectory timing and rewards](evidence/pace/pace-oracle.json)

**Parent launcher ready.** Six paired starts, four methods, one Baige Distributed Training RTX 4090 or RTX PRO 6000D with a 90-minute scheduler limit. The GPU driver replans every five controls on each branch's own two RGB images and proprioception. It reports every outcome and conditions the transfer count on matched OSC success. The CPU fixture verified the OpenPI input conversion, paired openings and real 7D/8D commands using archived proposals. GPU use here was zero. [Pinned launcher](launch-pinned.sh) · [Rendered fixture](evidence/fixture/live-summary.json) · [Six-start panel](evidence/panel/panel.json) · [Runtime pins](evidence/runtime.json)

The exact command from any directory on the allocated host is

```sh
p=7518022c88711815c8207b4397e4ce84036c6134
r=frozen-vla-native-controller-20261123-r1
bash "/mowenjie/fluffy-bassoon-2/outputs/$r/src-$p/outputs/$r/run.sh" "$p" gpu01
```
