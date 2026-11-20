# Goal 5 on-body demonstrations

The qualification asks whether an analytic controller can push the plate into the stove-front region on both source bodies at the same two released starts. A pass requires four accepted tapes, each ending in positive native reward and native success. A failed body/pose stays in the denominator. The coordinator owns the first Linux CPU qualification. Runtime success counts are pending.

## Expert and simulator

The expert reads the compiled world pose of `plate_1`, the `main_table_stove_front_region` goal site, the table surface and the collisions belonging to the plate, both fingers and the cream cheese. It closes the common gripper, aligns the tool above the trailing rim, lowers the fingertips into the rim's height range and pushes. Each 20 Hz command uses the current plate position, its measured radius, the lowest finger vertex and the tool axis. The first route aims inside the far side of the goal. A slip or prolonged stall triggers a lift and reseat. If the route fails, a second run from the same row and seed routes around the cream cheese. Simulator setup or integrity errors terminate that body/pose with an error record. All post-reset motion comes from native OSC actions. The expert uses geometric feedback for its commands and the released BDDL predicate for acceptance. [Expert](demos.py) · [Simulator](sim.py)

| Interface | Previous cross-arm run | This collector |
| --- | --- | --- |
| Reset | `ArmEnv(arm, "libero_goal", 5).reset(init_state_id, seed)`, same-seed Panda donor and named scene transfer on alternate arms | Calls that reset, retains its scene and model fingerprints, uses `seed_base + pose` on every body |
| Bodies | Panda, Sawyer, UR5e with the common PandaGripper | Also registers stock IIWA for the reserved evaluation. Collection accepts Panda/Sawyer and optionally UR5e development. |
| Action source | Frozen π0.5 proposed policy actions | Script computes seven bounded native OSC/gripper components from live geometry |
| Observations | Two native 256 × 256 views and public pose/finger sensors; previous motor had its own image and state preprocessing | Native agentview and wrist arrays in the same orientation as the simulator API; XYZ, hand quaternion XYZW, signed jaw opening |
| Response | Measured public displacement, world rotation vector and jaw change | Same seven measured components, in metres, radians and metres respectively |
| Outcome | Released task reward, native success and manifest horizon | Same step result. The BASE Goal 5 horizon is 300 controls. |

The previous simulator supplies the registration metaclass, LIBERO placement properties, stock XML and gripper identity checks, camera attachment checks, OSC bounds and reset admission checks. The session adds the IIWA XML digest to its in-process identity table. The source snapshot and run artifacts stay in the previous directory. The previous integration tested Panda/Sawyer/UR5e on Linux and verified Panda parity. This expert has its own qualification. [Previous simulator contract](../libero-crossarm-20261120-r1/SIM.md) · [Previous execution](../libero-crossarm-20261120-r1/EXECUTION.md)

| Stock robot | Arm joints | SHA-256 of wheel MJCF |
| --- | ---: | --- |
| Panda | 7 | `51e3a4a6d54bcaf77e9d36a27372b871728590eedf53d4fd0e195525950eb988` |
| Sawyer | 7 | `a0a9743d868c23e157e57b3415c582c2223379aefaebf54c5dbff3653998fbaf` |
| UR5e | 6 | `1c9818566a7bbba89a2c848523e6e1eb72db204d91b8a05d0bff5bbb2a4ad316` |
| IIWA | 7 | `7eb9d87f431f7fbfcb0e028e608956aa2c2f7dc456abb0dfb6f234fd8044a5dd` |

The runtime uses robosuite 1.4.0, MuJoCo 3.2.3 and LIBERO-Pro2.0 `e5500c690c1498eca1b7050350654964190cdf70`. The robosuite wheel SHA-256 is `aba065e7b36745738cede259457b2cb349427f3608728d867ef3a2034cb62994`. The shared PandaGripper MJCF SHA-256 is `d9d68123909d6910488310d0fae4e358ed4ac831c6d0957f3e30893a5412baac`. Static inspection matched the extracted robot XMLs to the wheel, checked their registration, joint counts, cameras and referenced wheel assets, and checked the BASE BDDL goal and horizon.

## First CPU qualification

Run from the verified project worktree root on the assigned Linux CPU. These commands use the previous CPU setup's Mesa EGL device 0. Choose a fresh attempt name for each launch. Keep development collection in its own output root so the motor's source directory contains only Panda and Sawyer archives.

```bash
set -euo pipefail
code=$PWD
root=/mowenjie/fluffy-bassoon-2
name=libero-paligemma-crossarm-20261120-r1
here="$code/outputs/$name"
setup="$root/setups/libero-meta-learning-20261117-r1"
export CROSSARM_RELEASE="$root/sources/libero-pro2-e5500c69"
export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export CROSSARM_SIM_PYTHON="$setup/env/bin/python"
export PYTHONPATH="$code/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$here"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBGL_ALWAYS_SOFTWARE=1 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
qual="$root/outputs/$name/cpu-qual01"
mkdir -p "$root/outputs/$name"
mkdir "$qual"
timeout --kill-after=15s 15m "$CROSSARM_SIM_PYTHON" -Bu "$here/demos.py" \
    --output "$qual/demos" --poses 0,1 --bodies Panda,Sawyer --task-id 5 \
    2>&1 | tee "$qual/qualification.log"
```

The collector creates a fresh output directory and refuses reuse. Each accepted file is `<Body>/<pose>.npz`. Load it with `allow_pickle=False`. For each of the T issued actions, `rgb` and `wrist` are uint8 [T,256,256,3], `state` is float32 [T,8], and `prompt` is a scalar Unicode instruction. Inputs are sampled before the action. The float32 [T,7] `actions` are native commands in [-1,1]. The paired float32 [T,7] `motion` and float32 [T] `rewards` come from the resulting step. The tape also records the final public `terminal_state`, body, pose, seed and scalar `native_success`.

`index.json` contains the command, source hashes, attempts, reset and scene lineage, renderer, route, first/final geometry, measured contact counts and accepted filenames. `quality.json` counts requested and accepted starts, attempts and first-pass successes by body. Failed routes write their commands, public states, motion, native rewards and pre-action geometry to `failures/*.json`. A shortfall exits with code 1 after recording the requested starts. When diagnosing a miss, compare the measured finger height to the rim, the contact count and the plate's travel relative to the goal.

Run this readback on the CPU host after qualification. It checks the accepted controls and pairing, and independently reconstructs every measured motion from the saved public states, including the final transition.

```bash
"$CROSSARM_SIM_PYTHON" -B - "$qual/demos" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

root = Path(sys.argv[1])
rows = json.loads((root / "index.json").read_text())["attempts"]
accepted = {(r["body"], r["pose"]): r for r in rows if r["accepted"]}
assert set(accepted) == {(b, p) for b in ("Panda", "Sawyer") for p in (0, 1)}
for (body, pose), r in accepted.items():
    info = r["reset"]
    assert info["model_identity"]["arm"] == body
    assert "llvmpipe" in info["renderer"][1].lower()
    paired = accepted["Panda", pose]["reset"]["environment_reproducibility"]
    for key in ("seed", "released_row_fingerprint", "scene_fingerprint"):
        assert info["environment_reproducibility"][key] == paired[key]
    with np.load(root / r["file"], allow_pickle=False) as a:
        n = r["steps"]
        assert n > 0 and bool(a["native_success"]) and r["native_success"] and r["done"]
        assert a["prompt"].dtype.kind == "U" and a["prompt"].item().strip()
        for key in ("rgb", "wrist"):
            im = a[key]
            assert im.shape == (n, 256, 256, 3) and im.dtype == np.uint8
        for key, width in (("state", 8), ("actions", 7), ("motion", 7)):
            x = a[key]
            assert x.shape == (n, width) and x.dtype == np.float32 and np.isfinite(x).all()
        assert np.max(np.abs(a["actions"])) <= 1
        rewards = a["rewards"]
        assert rewards.shape == (n,) and np.all(rewards[:-1] == 0) and rewards[-1] == 1
        x = np.vstack((a["state"], a["terminal_state"]))
        rot = Rotation.from_quat(x[:, 3:7])
        world = (rot[1:] * rot[:-1].inv()).as_rotvec()
        delta = np.column_stack((np.diff(x[:, :3], axis=0), world, np.diff(x[:, 7])))
        np.testing.assert_allclose(a["motion"], delta, atol=2e-6, rtol=1e-5)
        print(body, pose, r["route"], n, float(rewards[-1]), r["contact_steps"])
PY
```

## Session and process boundary

`Session(body, task_id=5)` runs natively when invoked with the simulator Python. From the VLM environment set `CROSSARM_SIM_PYTHON` to the pinned simulator interpreter and keep the release and simulator configuration above. A private, persistent subprocess uses a local pipe. Arrays cross it as raw camera bytes and numeric lists. Both paths return the same four public observation fields. Inspect `session.reset_info` for lineage. The expert reads privileged geometry through a separate method. Always close the session.

For a direct-versus-pipe check, run the following command in the same CPU qualification shell, with `VLM_PYTHON` set to the coordinator's VLM executable. It compares the pipe reset images and first transition against the direct native tape on the same source start.

```bash
"${VLM_PYTHON:?set to the VLM interpreter}" -B - "$qual/demos/Panda/0.npz" <<'PY'
import sys
import numpy as np
from sim import Session

with np.load(sys.argv[1], allow_pickle=False) as a, Session("Panda", task_id=5) as s:
    obs = s.reset(int(a["pose"]), int(a["seed"]))
    assert obs["prompt"] == a["prompt"].item()
    for key in ("rgb", "wrist", "state"):
        np.testing.assert_array_equal(obs[key], a[key][0])
    obs, motion, reward, done, success = s.step(a["actions"][0])
    np.testing.assert_array_equal(motion, a["motion"][0])
    assert reward == a["rewards"][0] and done == success == (len(a["actions"]) == 1)
    if len(a["actions"]) > 1:
        for key in ("rgb", "wrist", "state"):
            np.testing.assert_array_equal(obs[key], a[key][1])
PY
```

For production source collection, keep the same CLI and release, select the coordinator's source pose bank, and switch to the assigned Distributed Training container's qualified NVIDIA EGL environment. IIWA stays reserved for the gated evaluation.
