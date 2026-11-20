# H1 native command-frame collection

The registered collection has **256 freshly reset simulator trajectories, 4,096 measured transitions and 40 distinct donor layouts**. Both arms and every setting in a split replay the same donor and public probe for a given role and index. Each support and its paired query start on different layouts. The assignment below keeps every donor and all of its trajectories in one split. Design base `0fbc5e3167b362bc33b741e6bbfa20724aa93234`. [Registered design](../embodiment-meta-factory-20261120-r1/SCALE.md) · [Collector](collect.py)

| Split | Settings on both Panda and Sawyer | Indices per role | Support seeds | Query seeds | Tapes |
| --- | --- | --- | --- | --- | ---: |
| Train | `nominal`, `yaw_p90`, `yaw_m90`, `gain065` | 0–11 | 960000–960011 | 960100–960111 | 192 |
| Eval | `yaw_p45`, `yaw_p90_gain065` | 0–7 | 970000–970007 | 970100–970107 | 64 |

Each index offsets the first seed in its range. R2 samples the native BASE Goal 5 BDDL, checks contacts and excludes the 50 public Panda state fingerprints. The collector also requires each donor's raw state, complete scene, movable world XY and free-joint XY hashes to be unique across the 40 draws. It saves the flat donor state and fixture receipt in `collection.json`. Each tape calls R2's generated reset again. R2 rebuilds the executing arm, transfers the donor scene to Sawyer by bounded IK, applies ten native settling controls, and verifies the pre-settle scene and settled contacts. Within a family and block, the collector checks that the settled simulator state, settled scene and public opening match across settings. [R2 placement API](../libero-paligemma-dualarm-20261120-r2/generated_layouts.py) · [Camera-free session](../libero-paligemma-dualarm-20261120-r2/expert.py)

## Probes and preflight

The native controller receives `gain × Rz(yaw) × public_xyz` in world axes. Positive yaw maps +X toward +Y. Gain 0.65 applies to all three translation coordinates. The other four controls pass through directly. Each call advances the native 20 Hz relative OSC once. The label is the measured difference between consecutive public world grip-site positions, in metres. The collector checks it against the native transition packet. It preserves native rewards and termination flags.

The first eight tapes use train support index 0, one per family and training setting. They count toward the 192 training tapes. Every cell receives the same eight public pulses, each held for two controls. The order is +X, −X, +Y, −Y, +Z, −Z at magnitude 0.39, then +X, −X at 0.60. Angular commands are zero and the jaw command is −1. The gate uses the second measured displacement of each pulse so a change of command has one control period to take effect.

| Training setting | Expected world axes for public +X, +Y, +Z | Public inverse pair for nominal ±0.39 world X |
| --- | --- | --- |
| `nominal` | +X, +Y, +Z | ±0.39 X |
| `yaw_p90` | +Y, −X, +Z | ∓0.39 Y |
| `yaw_m90` | −Y, +X, +Z | ±0.39 Y |
| `gain065` | +X, +Y, +Z | ±0.60 X |

1. All eight measured projections must have the expected sign. Zero fails.
2. Both inverse commands must produce the nominal ±0.39 X issued translation within `1e-7` per component, allowing float32 roundoff.
3. Compare the measured world response to the same arm's nominal pair. For each pair, take half the positive-minus-negative displacement to remove constant drift. In each shifted setting, the inverse pair must be strictly closer to nominal in Euclidean metres than the original ±0.39 X pair. The nominal tape supplies the reference.

Each remaining block uses NumPy's default random generator with `probe_seed = layout_seed + 2000000`. It draws 16 independent six-component controls uniformly from [−0.4, 0.4] and a jaw command of ±1 at each step. Neither arm nor setting enters this draw. A failed reset, duplicate layout, mismatched opening, early episode end, sign failure or inverse failure stops the collection at that block. The script makes no replacement draws.

## Run on allocated Linux CPU

From the deployed checkout, use the qualified R2 simulator interpreter with robosuite 1.4.0 and MuJoCo 3.2.3. The command below uses the R2 shared-storage paths and camera-free CPU environment. Set `H1_OUT` to a fresh attempt directory. [R2 CPU environment](../libero-paligemma-dualarm-20261120-r2/cpu_smoke.sh)

```bash
set -eu
code=$(git rev-parse --show-toplevel)
shared=/mowenjie/fluffy-bassoon-2
setup="$shared/setups/libero-meta-learning-20261117-r1"
run="$code/outputs/embodiment-command-frame-20261120-r1"
export CROSSARM_RELEASE="$shared/sources/libero-pro2-e5500c69"
export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export CROSSARM_SIM_PYTHON="$setup/env/bin/python"
export PYTHONPATH="$code/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT"
export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
"$CROSSARM_SIM_PYTHON" -Bu "$run/collect.py" --out "${H1_OUT:?set a fresh shared output directory}"
```

## Reader handoff

`tapes.jsonl` is published after all 256 tapes pass collection checks. Each line is a JSON object keyed by `(family, setting, role, index)`. Its `commands` array is 16 × 7 public controls **before** the hidden mapping. Its `motion` array is 16 × 3 measured world wrist displacements in XYZ order. These are the two reader arrays. Pair support and query by index within family and setting. For the wrong-support control, swap the held-out setting while preserving family, role and index.

Audit fields include `split`, `layout_id`, `layout_seed`, `probe_seed`, `probe_kind`, placement hashes, 16 native `issued_actions`, 17 `wrist_world_m` observations, rewards, success flags and the R2 reset receipt. The pulse script is fixed and its recorded probe seed identifies the block. `collection.json` records the checkout revision, collector digest, donor evidence, returned control count and numeric preflight checks. On a handled failure it records the active block and error. Completed rows remain in `tapes.jsonl.partial`. Only the final JSONL supplies the two narrow banks and nested broad bank.
