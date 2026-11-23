# Baige Distributed Training handoff

From the verified detached factory checkout, set CODE to its absolute path and PIN to its full deployed Git SHA. Start one exclusive RTX 4090 or RTX PRO 6000D worker with 12 CPUs, 48 GiB, a 115-minute scheduler limit and zero automatic retries. Single GPU allocation is capped at 1.92 hours. The file evidence/palette.json must be included in the pinned commit. The parent orchestrator owns launch and cumulative accounting.

```bash
bash "$CODE/outputs/frozen-vla-visual-adapter-20261123-r1/launch.sh" "$PIN" gpu01
```

The first completed raw episode is a real native LIBERO evaluation with the frozen π0.5 checkpoint. Check the GPU receipt, launch SHA and that episode, then inspect progress every 20 minutes. The launcher writes to /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-visual-adapter-20261123-r1/gpu01 on shared NVMe. Each completed episode writes a JSON file. result.json requires all 18. A nonzero exit or an absent report leaves the panel incomplete. Never reuse an existing attempt directory.

For an independent CPU recheck set the pinned checkout paths used by launch.sh, export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 and __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json. Use the same CROSSARM_RELEASE, LIBERO_PRO_ROOT, LIBERO_CONFIG_PATH and PYTHONPATH as the launcher. Use a fresh output directory.
```bash
run="$CODE/outputs/frozen-vla-visual-adapter-20261123-r1"
sim=/mowenjie/fluffy-bassoon-2/setups/libero-meta-learning-20261117-r1/env/bin/python
motor=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
"$sim" -Bu "$run/qualify.py" "$OUT"
"$motor" -Bu "$run/qualify_ipc.py" "$OUT" "$OUT/palette.json"
```

The CPU experiment used commitless Git-blob transfer (git pack-objects, unpack-objects and cat-file) for the new scripts and loaded all existing simulator dependencies from the already deployed detached checkout 1068ed0c5. Input blob IDs and dependency revision are in evidence/. Neither CPU test loads π0.5 weights. [Question and protocol](PLAN.md) · [Measured Linux controls](RESULT.md)
