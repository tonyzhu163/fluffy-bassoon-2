# Parent submission handoff

**Ready for the two separate Baige Distributed Training submissions.** The launch revision `cb246fffcec1d620603a7da841d29199568fcfa0` is deployed and verified in a detached checkout on shared storage. The design was frozen at `06013c21cda1bc6a2a78eb2a92239dd7006167f0`. On Baige CPU, both compiled target arms accepted two distinct five-control chunks under raw, planar and full wiring on the same task, with matching physical resets, camera inputs, noise and opening proposals. The expanded CPU check covers the remaining registered starts. The parent owns GPU submission and monitoring.

**Scheduler settings.** Copy the native job `job-xaugtearcacf` in Baige 分布式训练. Reuse its PyTorch image, environment, shared filesystem mounting /mowenjie, exclusive single-card resource specification and one instance. Select one RTX 4090 or RTX PRO 6000D and set a one-hour scheduler wall limit for each job. Use the corresponding command below verbatim. The two jobs can run independently. Each arm admits only one attempt directory. The launcher times out its entire job step after 3,540 seconds, with a five-second kill grace period. The scheduler limit also bounds time outside the launcher.

**IIWA** · proposed job name `frozen-vla-native-iiwa-r1`

```bash
bash /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-multiarm-confirmation-20261123-r1/src-cb246fffcec1d620603a7da841d29199568fcfa0/outputs/frozen-vla-multiarm-confirmation-20261123-r1/run.sh cb246fffcec1d620603a7da841d29199568fcfa0 IIWA
```

**UR5e** · proposed job name `frozen-vla-native-ur5e-r1`

```bash
bash /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-multiarm-confirmation-20261123-r1/src-cb246fffcec1d620603a7da841d29199568fcfa0/outputs/frozen-vla-multiarm-confirmation-20261123-r1/run.sh cb246fffcec1d620603a7da841d29199568fcfa0 UR5e
```

Both jobs import the existing locked native runner directly from the clean `a8a997e7d06bb131d3f9710204c58e106f0f0904` checkout used by the template job. They source its env.sh and use its OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`, pi05-libero-openpi-20261027-r1 checkpoint, simulator Python, cv2 overlay, LIBERO-Pro2.0 commit `e5500c690c1498eca1b7050350654964190cdf70`, LIBERO config, desktop libraries and NVIDIA EGL vendor JSON. The evaluation GPU probe requires one scheduler-attested card. JAX requires one CUDA GPU. The EGL lookup resolves CUDA ordinal zero to the matching EGL device. Each native episode verifies the NVIDIA renderer and expected model. The locked matrices, source-fit digest and the 16 arm/task support tapes are checked across the two jobs. [Runtime bindings](runtime.sh) · [GPU checks](job.sh) · [Frozen design](PLAN.md)

The shared output directory is `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-multiarm-confirmation-20261123-r1/`. Check `gpu01-IIWA/` and `gpu01-UR5e/` for gpu.json, runtime.txt, one JSON per completed episode and summary.json after full readback. For the first pair check the raw and planar receipts on libero_10/5, including matching physical reset, raw public cameras, motor inputs, noise and opening proposal. Continue at 20-minute intervals until exit. Each successful arm supplies 24 native task episodes across eight layout-14 identities. Monitor the scheduler's allocated GPU time separately from the launcher's measured one-card elapsed seconds.

At a time limit, retain the completed JSONs. For a partial native table on Baige CPU, replace ARM with IIWA or UR5e below. The command reads existing outcomes in just that arm's directory. A missing episode stays missing.

```bash
arm=IIWA
folder=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-multiarm-confirmation-20261123-r1
here="$folder/src-cb246fffcec1d620603a7da841d29199568fcfa0/outputs/frozen-vla-multiarm-confirmation-20261123-r1"
code=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-native-transfer-20261123-r1/src-a8a997e7d06bb131d3f9710204c58e106f0f0904
native="$code/outputs/frozen-vla-native-transfer-20261123-r1"
source "$native/env.sh"
export PYTHONPATH="$native:$PYTHONPATH" NATIVE_HISTORICAL="$native/historical.json"
export NATIVE_MOTION_DIR=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/cpu01
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$python" -Bu "$here/report.py" "$arm" "$folder/gpu01-$arm" --partial
```
