# Pinned Baige reproduction

The run belongs to `outputs/frozen-vla-gripper-20261123-r1`. It ran on the existing Baige development host via `baige-dev` with CPU affinity 18,19, software EGL, an empty CUDA mask, single-threaded BLAS and no scheduler job. The launcher creates a fresh attempt directory on the project share and refuses a reused name or a dirty/mismatched checkout. Both stages used the existing locked simulation environment, robosuite 1.4.0, MuJoCo 3.2.3 and LIBERO-Pro2.0 release `e5500c690c1498eca1b7050350654964190cdf70`. Both source and target policy collectors loaded the frozen OpenPI JAX `pi05_libero` checkpoint at `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1`. Source layout and noise seeds remain in the evidence. These CPU stages execute previously recorded policy commands or declared diagnostic pulses and do not load or optimize the policy.

The following exact commands rerun the two completed experiments sequentially in their respective immutable Baige checkouts. They require the two original source tapes and the historical native supports on the share. Use previously unused names for subsequent repetitions.

```bash
ssh baige-dev 'p=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-gripper-20261123-r1; "$p/source/outputs/frozen-vla-gripper-20261123-r1/run.sh" 1189be4fc82dc4d9eba3ef454772c8797406696e cpu-repeat01'
ssh baige-dev 'p=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-gripper-20261123-r1; "$p/source02/outputs/frozen-vla-gripper-20261123-r1/run.sh" c30e7f76989563a808fc5985b0679d18a5066e4c fork-repeat01 grasp'
```

For the parent repository, the incremental `handoff.bundle` is rooted at `a8a997e7d06bb131d3f9710204c58e106f0f0904`. It contains only commits adding paths under this run. To inspect its tip without touching a checkout or branch:

```bash
b=outputs/frozen-vla-gripper-20261123-r1/handoff.bundle
git bundle verify "$b"
git fetch "$b" refs/heads/frozen-vla-gripper-20261123-r1
git diff --stat a8a997e7d06bb131d3f9710204c58e106f0f0904 FETCH_HEAD
```

[Mechanistic results and decision](RESULT.md) · [Initial design](PLAN.md) · [Grasp fork and source-only selection](GRASP-PLAN.md)
