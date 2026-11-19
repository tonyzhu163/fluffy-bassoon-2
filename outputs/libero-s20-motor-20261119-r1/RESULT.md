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
separate seven-control queue to commands. Under the verified post-action
convention, the command-to-wrist alignment peaked at 14 controls in all 60
S20 traces and one control in all 60 BASE traces.
On Baige CPU we reverified all four cached HDF5 hashes and counted **zero
exact all-axis arm waits across 15,732 S20 controls** in 60 successful
traces. Goal 4 median length rose from 167 to 288 controls and Goal 8 from
154 to 232.5. This rules out literal zero-arm waiting as the recorded
strategy in these traces. Small commands can still produce little motion.
The upstream documentation identifies a privileged-state oracle. The stored
metadata gives its final phase and reports retries in three of the 60 S20
episodes. The oracle's state-to-action procedure remains unverified.
[Command summary](command-summary.json) · [Alignment and oracle metadata](demonstration-alignment.json) · [Training guide](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/docs/training.md) · [Simulator timing implementation](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/nonphysical_perturbations.py)

## Native physics replay

Replaying the eight preselected command sequences from their saved reset
IDs and seeds reached native success within the recorded command sequence
in **4/4 BASE and 2/4 S20** episodes. Every recorded demonstration had
originally succeeded. All five first recorded proprioceptive arrays
matched the simulator exactly **after issuing the first saved action** in
all eight replays. The reset comparison used in the first pass was one
command early. Pair stored observation `t` with `actions[t + 1]` for a
causal action target. The archive's first action lacks a stored preceding
observation.

With that alignment, peak wrist error was 0.003 mm on BASE and 7.3 cm on
S20. The check reproduced some successful delayed actions and still found
later S20 state divergence. Use the logged pairs as imitation supervision
on their recorded observations and evaluate the adapted motor in fresh
closed-loop runs. [Per-episode evidence](replaycpu02.json)

Baige `replaycpu02` used `01841848`, the released environment adapter at
`e5500c69`, and the existing MuJoCo runtime with camera sensors disabled.
All eight episodes finished in 49 wall-clock seconds, with zero GPU-hours.
The earlier pass at `bbca9415` gave the same success outcomes. The rendering
attempt was withdrawn while waiting for the shared lock.
