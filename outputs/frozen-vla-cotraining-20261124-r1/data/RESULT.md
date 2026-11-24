# Successful source demonstrations for native OpenPI flow training

The staged dataset contains 192 training windows from 21 successful Panda trajectories and 64 read-only validation windows from eight different successful Panda trajectories. Every source task contributes 24 training and eight validation windows. The two files have disjoint full-trajectory state/action hashes. Each example has two 256 × 256 RGB arrays, native public proprioception and instruction, four slots of masked causal history, and ten consecutive seven-dimensional expert commands. Overlapping windows from one demonstration share a split. The statistical unit for source validation is the trajectory.

## Exact split

Numbers are episode indices at Physical Intelligence release `a4336d589d589045d1c56423ffdf3b88a0e19b1f`. The manifest lists each zero-based decision index, upstream RLDS shard and record ordinal, source hash and task prompt. [Manifest](manifest.json) · [Prospective source selection](selection.json)

| Source task | Train episodes | Held-out episode |
| --- | --- | --- |
| Object · ketchup into basket | 808, 809, 815 | 820 |
| Object · alphabet soup into basket | 813, 818 | 825 |
| Object · barbecue sauce into basket | 811, 812 | 824 |
| Object · butter into basket | 819, 827 | 828 |
| Spatial · bowl next to cookie box onto plate | 1261, 1274, 1277 | 1291 |
| Spatial · bowl in top drawer onto plate | 1262, 1263, 1268 | 1276 |
| Spatial · bowl on ramekin onto plate | 1264, 1265, 1266 | 1271 |
| Spatial · bowl between plate and ramekin onto plate | 1272, 1273, 1275 | 1282 |

The registered episode IDs were selected by task frequency in the initial released LeRobot episode range for each source suite, breaking ties by prompt. Up to four episodes in ascending order contributed to each task. Its last selected episode went to validation. No random seed or model outputs enter selection. Window starts span each trajectory up to its last complete ten-command target. The earliest sampled decision is episode index modulo four, which supplies examples at reset and all three partially filled buffer ages.

## Lineage and transfer boundaries

The official LIBERO PFS mirror contained download metadata and empty task directories, and the earlier reachability packs retained simulator state and terminal labels while omitting both RGB streams. The source used here is OpenVLA's release of Panda expert demonstrations replayed in LIBERO. Its generator executes the archived expert commands, collects both public images and proprioception immediately before each executed command, and saves the trajectory when the final simulator step reports success. It excludes no-op actions before stepping. All 29 selected RLDS records have terminal reward one with the terminal and last bits set. Physical Intelligence converted these replays to LeRobot. Every state and issued control in each selected LeRobot episode matched the corresponding RLDS sequence exactly at float32 precision. The success evidence comes from the archived native replay. We did not rerun the trajectories in a simulator. [Original reachability archive notes](../../frozen-vla-reachability-20261123-r1/RUN.md) · [Replay and success filtering](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/regenerate_libero_dataset.py) · [Pinned RLDS release](https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551) · [Pinned LeRobot release](https://huggingface.co/datasets/physical-intelligence/libero/tree/a4336d589d589045d1c56423ffdf3b88a0e19b1f)

Both splits contain exclusively Panda, LIBERO-Object and LIBERO-Spatial. The parent native panel uses LIBERO-Goal tasks 3 and 6 and LIBERO-10 task 8, so every registered Kinova3 start and Panda retention start is excluded by task identity. Source validation lives in its own HDF5 file. Restrict fitting, calibration and normalization estimation to the training split. The locked stock model has its own LIBERO pretraining exposure. The separate fitting-task experiment must establish scene identity against its own registered source-task starts because those tasks overlap by design. [Native panel](../evaluation/PANEL.md) · [Fitting-task protocol](../matched-task/PLAN.md)

## Camera, state and history checks

The RLDS conversion rotates both native cameras through 180 degrees. The primary LeRobot release stores that orientation in PNGs. The reader returns native orientation to the existing converter, which rotates and resizes for OpenPI. All 512 resulting 224 × 224 RGB arrays matched the direct path from decoded release PNG through the stock resize and uint8 operations, pixel for pixel. As a separate check, comparing the two archives at 256 × 256 gave a mean absolute RGB difference of 1.0775 levels for aligned images and 69.8575 for the 180-degree alternative. The RLDS source uses JPEG. The staged pixels come from the published LeRobot PNGs. [RLDS conversion](https://github.com/moojink/rlds_dataset_builder/blob/6174b0b6bb69df6361f1117944952bf14afb0cc3/LIBERO_Object/LIBERO_Object_dataset_builder.py) · [OpenPI LIBERO conversion](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/convert_libero_data_to_lerobot.py) · [Released live camera path](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py)

The source stores wrist axis-angle. Reconstructing XYZW and running the stock observation converter recovered all 256 eight-value released states exactly after casting to float32. Maximum absolute error before the cast was 8.89 × 10⁻¹⁶. The 256 action chunks matched the direct ten-row slices, and every history vector matched an independent reconstruction from the preceding issued controls, XYZ and finger joints. Both splits exercised history depths zero through four. Reset, oldest-slot eviction and snapshot independence passed. These checks ran with CUDA hidden and affinity restricted to Baige CPUs 24 and 25. [Full indexing receipt](smoke.json) · [Reader API](API.md)

## Staged files and reproduction

Baige shared project directory `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1/data`.

| File within that directory | SHA-256 |
| --- | --- |
| `dataset-v1/manifest.json` | `05209732b5e3f04308c8acbef2f6c8c8a2f173673840fdb055e1b094ff7d6f40` |
| `dataset-v1/train.h5` | `eb053669d1c61cdc252f48775f0311739bd86dbdb117ea75d38fee96554a0843` |
| `dataset-v1/heldout.h5` | `bf8cba16d4f8f29d9f6e8c5eebc2c0ef290f9550d15628168abadaf1b60e2a13` |
| `smoke-v1.json` | `7e2163cb2670239fffd2c38f57baf0af13056ea97efaf37bc7d23a3dfee22689` |

At stage time the GPFS project mount reported 139 GiB available and a write probe succeeded. The host exposes filesystem capacity and no separate quota CLI. Source inputs occupy 858 MiB and the staged files occupy 97 MiB. Build and validation used the existing uv OpenPI environment. The immutable data-code checkout is commit `97bedc866757c5004fdd4bc24fb5006662bbbd82`. From the Baige project directory above, these commands rebuild into a fresh destination and index every example.

```bash
set -eu
d=$PWD
code=$d/code-97bedc866757c5004fdd4bc24fb5006662bbbd82
src=$code/outputs/frozen-vla-cotraining-20261124-r1/data
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=$d/tools/tfrecord-1.14.6/tfrecord:$code/src:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1
taskset -c 24,25 "$py" "$src/preprocess.py" "$d/inputs" "$d/rebuilt"
taskset -c 24,25 "$py" "$src/smoke.py" "$d/rebuilt" "$d/inputs" "$d/smoke-rebuilt.json"
```

For another machine, fetch each path listed in `archives.json` from the pinned RLDS repository and each path in `selection.json` from the pinned LeRobot repository. Place the files in a single input directory using their basenames. The preprocessor checks size and the release's SHA-256 for every input before conversion. It uses Python 3.11, NumPy, PyArrow, h5py, Pillow and the TensorFlow Example schema from `tfrecord==1.14.6`. The Baige overlay contains only that package's generated `example_pb2.py`. The source distribution SHA-256 is `03483dd1745b80239dc4510e22decc4c97aa5178704bdfa2d01d1636edf9f5c7`. Runtime readers require only NumPy and h5py. [Archive pins](archives.json) · [Preprocessing](preprocess.py) · [Independent checks](smoke.py)
