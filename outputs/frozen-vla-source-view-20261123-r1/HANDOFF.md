# Parent launch and reproduction

The Baige shared filesystem already contains a detached checkout with the qualified renderer, the ten lossless inputs, and the parent launcher. Request **one RTX 4090 or RTX PRO 6000D, 12 CPUs, 48 GiB, at most 60 minutes from allocation, zero retries** in Distributed Training. Use this job command, including the pin check. The launcher also limits its whole process group to 3,500 seconds with five seconds of kill grace. An absent pair or an expired job leaves the corresponding outcome unmeasured.

```bash
set -eu
pin=2b59cff7e173ee8064905256b98a1dd3fd0f79d5
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-source-view-20261123-r1
bash "$run/code-$pin/outputs/frozen-vla-source-view-20261123-r1/launch.sh" "$pin" gpu01
```

The script checks the Git revision, GPU count/type, release and OpenPI revisions and input checksums. All policy calls use one loaded `pi05_libero` checkpoint, the normal image preprocessing and the stock noise schedule. For each CPU frame the call order is untouched, Kinova render, Panda render, untouched repeat. Each `proposals-<layout>-<control>.json` contains the input and noise hashes, all four 10 × 7 proposal arrays, paired first-action differences, per-dimension chunk RMS and the L2 difference over the first five bounded actions. Compare Panda minus Kinova to both same-input contrasts at that same frame. The saved camera hashes identify which views changed. Frames on one trajectory are dependent.

Six requested native episodes use Kinova, the original task instruction and unmodified native OSC on starts 16 and 17. Each branch samples its next observation from its own physical trajectory after five controls. Each layout gets untouched, Kinova render and Panda render branches in one process with the same reset and noise. The simulator child owns the render siblings. Full re-reads of native RGB, clean scene RGB/depth and collision/IK admission run before the relevant image is exposed. If a later Panda pose fails, the renderer records the reason and the paired score is censored. Compare success within layout in `native-pairs.json`. `native-*.json` retains rewards, native stop conditions and counts. `calls-*.json` contains input, noise and proposal digests at every replan. `views-*.jsonl` records the geometry and pixel checks. Inspect the GPU receipt and the first completed native episode, then check at 20-minute intervals through exit. No outcome from a separate job is a control.

## Reproduce on two Baige CPU cores

Run each command on the development host with a fresh destination name. Both wrappers pin all children to CPUs 36 and 37 with GPU visibility disabled and software EGL. The first reads only the two Kinova tapes under the prior native run’s `gpu01/`. Their SHA-256 digests are in the per-layout results. The original CPU replay used the revision below and depended on native runner `a8a997e7d06bb131d3f9710204c58e106f0f0904`, LIBERO-Pro2 `e5500c690c1498eca1b7050350654964190cdf70`, MuJoCo 3.2.3 and robosuite 1.4.0.

```bash
set -eu
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-source-view-20261123-r1
name=frozen-vla-source-view-20261123-r1
cpu=dc956f5e3f2f6a88b817c6d33350d87464193c97
pin=2b59cff7e173ee8064905256b98a1dd3fd0f79d5
bash "$run/code-$cpu/outputs/$name/cpu.sh" replay01
bash "$run/code-$pin/outputs/$name/smoke.sh" wire01
```

The first command produces the ten lossless RGB/proprioception bundles, pose and replay readbacks, and display JPEGs. Check unpacked arrays when comparing a reproduction. The second uses the actual OpenPI image conversion on the committed frames, then exercises the production simulator transport using six specified CPU controls per branch. The fixed command generator supplies a physics and transport check only. The frozen-policy proposal comparison requires the parent’s GPU job.
