# Baige execution record

On November 20 the CPU development allocation `dv-63739b1202` exposed the pinned LIBERO and OpenPI environments. The project filesystem had 452 GiB free of 2 TiB and accepted a fresh write. The Pro6K distributed queue reported 97/416 GPUs free and 16/32 free under the personal quota when the job form was prepared.

The first stock-construction attempt stopped at EGL import because the software renderer on this CPU host exposes a single device numbered 0. Repeating on that device in robosuite 1.4.0 on fitting task `libero_goal/4` created a seven-joint `MountedPanda`. Its first zero command returned native reward 0. Stock construction of Sawyer and UR5e raised `KeyError('MountedSawyer')` and `KeyError('MountedUR5e')`. Runtime registration of mounted stock arm classes is required for this benchmark. This check took zero GPU-hours.

Raw logs are under `/mowenjie/fluffy-bassoon-2/outputs/libero-crossarm-20261120-r1/preflight{01,02}/preflight.log` on the project volume. The corrected run used the simulator environment from `setups/libero-meta-learning-20261117-r1`. The nominal preflight was a factory-construction check only and did not use a published evaluation start.
