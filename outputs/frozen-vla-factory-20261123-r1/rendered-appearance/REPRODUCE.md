# Reproduce the fixed Panda insert on CPU

Start at the parent appearance commit `8af213a478f0ca9ecafd1fc23035b2cfd32cd692` with the sibling run's scratch data. For a fresh installation, follow the [parent setup and renderer](../appearance/REPRODUCE.md) through the call to `render.py`. This supplies the two pinned Python environments, source assets, checkpoint, tokenizer and ten paired cases. The parent code pins LIBERO-Goal tasks and seeds, robot models, renderer, LeRobot and unnormalization. From this directory run the following with `WORK` set to the absolute scratch path.

```bash
set -eu
: "${WORK:?set WORK to an absolute scratch path}"
export LIBERO_CONFIG_PATH="$WORK/libero-config" PYTHONHASHSEED=0 NUMBA_DISABLE_JIT=1
export LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
out="$WORK/followup/report"
mkdir -p "$out"
"$WORK/sim/bin/python" fixture.py "$WORK"
"$WORK/sim/bin/python" rig.py "$WORK/followup/public" "$WORK/followup/sprites"
cp "$WORK/followup/sprites/installation.json" "$out/"
"$WORK/sim/bin/python" diagnose.py "$WORK/followup" "$out"
"$WORK/env/bin/python" evaluate.py "$WORK" 0 5
"$WORK/env/bin/python" evaluate.py "$WORK" 5 10
"$WORK/env/bin/python" summarize.py "$WORK" "$out"
```

The fixture builder reconstructs the earlier matched Panda and UR5e cases and asserts exact equality of both RGB frames and masks. It exports the UR5e public observation separately. The only inputs read by the rig are the target end-effector position, hand quaternion and two finger positions. The calibration environment uses a fixed camera, robot base, table and lights, with task objects hidden and the arm silhouette supplied by geometry IDs in the generated image. The public operation pastes exactly those sprite pixels onto the unmodified target camera image. It has no query object geometries, segmentation, background inpainting or target-arm erasure. The displaced comparison uses the same sprite translated by 128 raw-image rows with no clipping, wraparound or overlap with the original arms.

The evaluator canonicalizes the target quaternion into the hemisphere of the fixed rig's rest hand before calling the pretrained source motor. Opposite quaternion signs encode identical rotations but can yield different axis-angle representations at the processor. This reproduces the original source state at float32 input precision on these ten cases. The target wrist images also match the original wrist inputs byte for byte. All conditions use target state, target wrist, the same task text and the same explicit Gaussian tensor (CPU seed 9100 + task index). A repeated no-op forward checks all 350 entries exactly. The summarizer checks the complete source chunk against the saved parent hash and all no-op scores against the prior full-native control.

The evaluator reads oracle arrays to build the reference and upper controls only. The diagnostics compare the same rig rendered at inferred and true joint angles, then at true angles with the actual Panda scene visibility mask. `geometry.csv` reports silhouette IoU, missing visible pixels and RGB differences at shared pixels (0–255 channel scale). `scores.csv` holds all seven measured per-task conditions. The primary metric is RMS over 50 × 3 translation commands in unnormalized action units. `summary.json` includes paired differences in reference distance and direct distances between diagnostic chunks. Intervals are the 2.5th and 97.5th percentiles from 20,000 whole-task resamples with equal weights and seed 20261123.
