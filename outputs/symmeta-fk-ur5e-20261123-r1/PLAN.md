# Does more accurate yaw improve UR5e task success?

On the locked UR5e × π₀.₅ benchmark, compare the published seed-41 visual yaw with forward kinematics (FK) plus planar alignment of the same three measured support displacements. Both receive the published SymMeta gain and gripper estimate. Trusted FK assumes correct joint sensing, robot geometry, and registration of the world/base and policy frames. The benefit of avoiding those inputs matters if the methods have similar native success.

The evidence available before launch is 1.93° versus 6.67° mean absolute yaw error across all 96 support traces and 2.02° versus 6.83° on the 64 held-out traces. An improvement in FK task success would support the hypothesis that the visual reader's residual yaw error costs tasks. Similar or lower FK success would weaken that explanation, subject to uncertainty across eight tasks.

## Paired evaluation

- Lock the official eight tasks × twelve transforms, original passive supports, query layout 13, seed schedule, policy weights and all other wrapper settings. Run both methods afresh and record their estimates, reset fingerprints and policy-noise fingerprints. Check support identities and exact pairing before pooling.
- Score native simulator task completion. Report both counts, paired FK wins and losses, success difference, task-level effects, the held-out subset and uncertainty across the eight task clusters. Show how the rerun compares with the historical SymMeta reference. A single panel cannot establish equivalence.
- Start with the qualified Baige Distributed Training runtime on RTX PRO 6000D. Target at most eight GPU-hours for this pilot, including launch recovery. Check actual GPU model, one real simulator episode, final allocation time, and exact code pin in the execution record. CPU checks carry zero GPU-hours.
- UR5e × π₀.₅ is the initial panel. Subsequent arm/policy panels need their own verified support and joint kinematics. The camera perturbation and demonstration comparisons are independent questions.

[Source evidence](../symmeta-camera-command-20261123-r1/RESULT.md) · [Published task and method rows](../xemb-live-wave-20261121-r1/evidence/pi05-rows.jsonl) · [Submission plan](../../papers/icml-xemb-sysid/TODO.md)
