# Baige execution record

On November 20 the CPU development allocation `dv-63739b1202` exposed the pinned LIBERO and OpenPI environments. The project filesystem had 452 GiB free of 2 TiB and accepted a fresh write. The Pro6K distributed queue reported 97/416 GPUs free and 16/32 free under the personal quota when the job form was prepared.

The first stock-construction attempt stopped at EGL import because the software renderer on this CPU host exposes a single device numbered 0. Repeating on that device in robosuite 1.4.0 on fitting task `libero_goal/4` created a seven-joint `MountedPanda`. Its first zero command returned native reward 0. Stock construction of Sawyer and UR5e raised `KeyError('MountedSawyer')` and `KeyError('MountedUR5e')`. Runtime registration of mounted stock arm classes is required for this benchmark. This check took zero GPU-hours.

Raw logs are under `/mowenjie/fluffy-bassoon-2/outputs/libero-crossarm-20261120-r1/preflight{01,02}/preflight.log` on the project volume. The corrected run used the simulator environment from `setups/libero-meta-learning-20261117-r1`. The nominal preflight was a factory-construction check only and did not use a published evaluation start.

The first integrated CPU pass at `2620e725` restored the same named joint and fixture scene on Goal 4, start 0. All three mounted arm XMLs and their 7/7/6 arm joints passed identity checks, both 256×256 cameras returned observations, and each model executed three controls. The observed first +X pulse produced 2.157 mm (Panda), 2.143 mm (Sawyer) and 2.351 mm (UR5e) of simulator grip-site X motion. Each used the same two-finger gripper. These are integration measurements. Reader inputs in the training revision are computed directly from successive public pose and finger observations. The privileged geometric readback remains available for reset auditing.

The fixed four-start native evaluation runs even if the source or target positive controls fail. The calibration checks constrain interpretation. They do not select checkpoints or exclude target starts. This choice was made before collecting policy outcomes.

Review before GPU submission moved all UR5e calibration to after saving and reloading the source-only checkpoint. Each query bank now receives the same practice commands and reset lineage across both labeled source bodies. The inverse reports the measured response rank and damps unsupported directions. The local inverse is a diagnostic comparator whose native score can be higher or lower than the learned adapter.
