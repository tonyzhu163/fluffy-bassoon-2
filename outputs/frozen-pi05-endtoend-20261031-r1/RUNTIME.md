# Frozen pi0.5 end-to-end runtime preparation — 2026-10-31

The existing Motus effc6 motor, RLinf c605/RoboTwin dca9 source and retained Python3.11/Torch2.11 cu128 runtime are bound in `RUNTIME-BINDINGS.json`. No shared environment or active run changed. The7.47GB checkpoint file is bound to its existing public-download receipt; the fresh allocated tensor hash, CUDA/parity/gradient and real episode remain pending.

`JOINT-CONVENTIONS.json` maps all12 arm coordinates to SHA-bound named revolute URDF joints. Literal simulator limits are−10/+10 radians, so the10%-of-range ceiling is2rad. This is not a manufacturer limit claim. Grippers6/13 are absolute normalized commands clipped to[0,1] by RoboTwin; the model decoder itself does not clip. Both grippers remain uncorrected.

For both candidate tasks the official controller-action horizon is400. The scalar reset issues zero decoded policy actions, so48 episodes/key imply19200 actions/key, with all six keys subordinate to the shared source120000 and phase300000 caps. Internal PhysX setup ticks are not separately instrumented or claimed zero; qualification must confirm the reset policy-action counters.

Storage/write preparation preserves a500GiB free floor after reserving this64GiB, task5764GiB and a concurrent128GiB allowance. This is a capacity snapshot, not an independent quota guarantee. No canonical CampaignBudget header or runtime reservation was created. The final executable/profile/arm-bound pins and focused allocated CPU qualification are the remaining admission inputs; later collection/student code does not gate A1.
