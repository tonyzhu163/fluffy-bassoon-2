# Frozen π₀.₅ on simulated IIWA and UR5e

Registered November 23, 2026, before accessing any new Kinova outcomes. The question is whether a full 3D linear translation correction, estimated from one short target trajectory, rescues more native failures than the planar correction and preserves inputs on which the frozen policy already succeeds. Failure to beat the planar correction or raw policy, losses on working inputs, or an absence of raw failures limits the corresponding claim. A lower motion-prediction error alone cannot answer this question.

Use all eight official held-out identities in the following fixed order on each arm. The table gives archived raw π₀.₅ native success on layout 13. These labels describe historical occupancy. Eligibility for rescue or harm is determined afresh by the paired layout-14 raw outcome.

| Suite | Task | IIWA layout 13 | UR5e layout 13 |
| --- | ---: | --- | --- |
| libero_10 | 5 | success | success |
| libero_10 | 8 | failure | success |
| libero_goal | 3 | failure | success |
| libero_goal | 6 | success | success |
| libero_object | 1 | success | success |
| libero_object | 8 | success | success |
| libero_spatial | 0 | success | success |
| libero_spatial | 8 | success | success |

Historical input is the [locked native archive](../frozen-vla-native-transfer-20261123-r1/historical.json), SHA-256 90acca7d4b5c872626000a55a06adf9f1758fa3086ec94c54b1d7ad1cfb5dc17. Include every identity, even when a historical failure disappears on a fresh start. UR5e has no historical raw failure in this panel.

Use released BASE layout 14 exactly once per identity and arm. Retain the locked runner's seed formula, 41800000 + 10000 × suite index + 100 × task ID + layout, with suite order as in the table. Use one policy-noise bank per case, native horizon, nominal world-frame relative OSC and the sim's binary task reward. No imposed shifts. Keep the released target's agent and wrist camera inputs and stock OpenPI preprocessing throughout. Use frozen pi05_libero, original quaternion convention, five-control replanning, and no source-policy trajectory. Compare raw, then planar, then full linear on the same target reset. The robot and gripper remain visible as rendered.

Use the fitted passive motion matrices (SHA-256 20a9d6800e10404ee23a542c70a3ba1fb077daeccd56cf7fea93555c7add64cf) from the [motion workstream](../frozen-vla-motion-models-20261123-r1/PLAN.md). The source prior is trained on Panda/Sawyer; each adapted condition shares the same single 15-control target trajectory on layout 11 for its arm and task. No fitting or tuning on query actions, rewards or images. Apply the supplied row-vector matrix to clipped XYZ only and clip to OSC bounds. Preserve rotation and gripper commands. Use compiled robot joint metadata for per-step tensor sizes. The common seven-component OSC command is independent of the number of physical arm joints.

Report a complete task table per arm, native successes and failures by method, the layout-13 to layout-14 raw occupancy table, and all four paired transitions for planar vs raw, full vs raw and full vs planar. Divide rescues by reference failures and harm by reference successes. When an eligible count is zero, give a null rate and the actual denominator. Report net success difference, both signs of discordance and uncertainty across the eight paired task identities. Keep arm-specific estimates, retain joint task identities across arms, and do not count methods or arms as independent replicates. Report elapsed and allocated GPU occupancy separately from task metrics. Compare architectures on native success only. The task set is fixed, with one layout and one noise draw per identity. Simulation scope only.

The light screen stops at 24 native episodes per arm (48 total). No second layout in this registration. Each arm gets at most one GPU-hour total on one Baige Distributed Training RTX 4090 or RTX PRO 6000D card. The parent submits one job per arm with a one-hour scheduler limit. Preserve partial receipts and report missing pairs if interrupted. Repeating or extending requires accounting for unused budget. Qualify actual compiled models on two Baige CPU cores with Mesa, checking paired native action wiring under all three methods. CPU fixtures supply no policy outcome.
