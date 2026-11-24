# Reproduce the Baige CPU test

Run from this directory on Baige with the existing project LIBERO environment. The launch wrapper selects Mesa llvmpipe, hides CUDA and pins the entire process to CPUs 24 and 25. The simulation uses MuJoCo 3.2.3, robosuite 1.4.0 and LIBERO-Pro2.0 release e5500c69. It imports the native reset interface and checks the previously recorded per-arm CPU reset against the native worker's Git pin a8a997e7. Set NATIVE_CODE if that checkout lives elsewhere. All command paths below are within the owner's run directory except the read-only native action input.

```bash
set -eu
here=$(pwd -P)
dest=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-reachability-20261123-r1/reproduction01
mkdir "$dest"
bash cpu.sh feasibility.py "$here/demos" "$dest" goal3 long8 goal6
bash cpu.sh verify_geometry.py "$dest"/goal?-1?.json "$dest"/long8-1?.json --out "$dest"
bash cpu.sh tilt.py "$dest"/goal?-1?.json "$dest"/long8-1?.json --out "$dest"
native=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-native-transfer-20261123-r1/gpu01
bash cpu.sh realized_drawer.py "$native" "$dest"
bash cpu.sh summarize.py "$dest" "$dest"
```

The three small HDF5s in `demos/` contain two groups per task and omit images. They retain recorded states, joint and tool observations, episode outcome, recorded joint order and fixture transforms. To recreate them from the original public HDF5s, use the commands below with an environment that contains NumPy and h5py. Full downloads total 3.62 GB. The packer checks the release SHA-256 before it copies any state. [Pinned original demonstrations](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a)

```bash
bash fetch.sh inputs
for task in goal3 long8 goal6; do
    python pack.py inputs "$task" demos
done
```

The archived evidence contains compressed JSON inputs to `summarize.py`. Recreating the three CSV tables from those records requires only Python 3. Reproducing the simulator and analytic joint-bound readbacks requires the Baige runtime above.

```bash
python3 summarize.py evidence "$dest"
```
