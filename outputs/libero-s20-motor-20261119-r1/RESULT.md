# Released S20 commands and fitting-data coverage

The pinned archive provides BASE and S20 on all 24 fitting tasks. The
publisher's copy filter retains 720 BASE and 691 S20 demonstrations, spanning
611 and 195 task/reset combinations respectively. The retained rows contain
152,722 BASE and 163,826 S20 recorded actions. LIBERO-10 task 4 accounts for
29 excluded S20 copies and has one six-control source. The existing paired
index found only 122 matching reset-ID blocks across ten tasks. They supply
supervision on visited states. Labels at states reached by the frozen
policy require a teacher queried at those states or simulator branches.
[Pinned training release](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro2.0-Train/tree/6c4c5524676815a88880a7025badd17fb724f7f4) · [Earlier paired index](../libero-pro2-demonstration-pairs-20261118-r1/RESULT.md)

The pinned benchmark applies a seven-control queue to observations and a
separate seven-control queue to commands. The earlier HDF5 audit measured
an effective 13-control command-to-wrist offset on both delayed bowl tasks.
On Baige CPU we reverified all four cached HDF5 hashes and counted **zero
exact all-axis arm waits across 15,732 S20 controls** in 60 successful
traces. Goal 4 median length rose from 167 to 288 controls and Goal 8 from
154 to 232.5. This rules out literal zero-arm waiting as the recorded
strategy in these traces. Small commands can still produce little motion.
The upstream documentation identifies a privileged-state oracle. Public
release files expose the successful action traces and standard teleoperation
collectors. The oracle's state-to-action procedure remains unverified.
[Command summary](command-summary.json) · [Training guide](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/docs/training.md) · [Simulator timing implementation](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/nonphysical_perturbations.py)
