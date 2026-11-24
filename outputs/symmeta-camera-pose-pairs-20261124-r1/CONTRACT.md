# Reader bank

Read the active bank path from `/tmp/frozen-adaptation-1124/camdata-contract` on the coordinating Mac. All NPZ paths refer to Baige shared storage. A bank contains `rows`, one per distinct robot trajectory. Filter `role == "heldout"` (UR5e) for evaluation. Historical-only queries additionally filter `original_orbit == true` and views nominal, orbit−, orbit+. The ASCII view keys are `nominal`, `orbit-`, `orbit+`, `roll-`, `roll+`. Each row supplies suite, task_id, seed, init_state_id, arm, native_control_id, index and source tape digest. Row indices are unique only within arm. The diagnostic labels are control_yaw_deg, xy_scale and gripper_sign. These and task identity must stay out of reader inputs.

```python
import json
import numpy as np
bank = json.load(open(bank_path))
row = next(r for r in bank["rows"] if r["id"] == "UR5e-0008")
v = row["views"]["orbit+"]
H = np.asarray(v["H_ref_to_view"])  # sample observed at H @ [col_ref,row_ref,1]
H /= H[2, 2]
with np.load(v["passive_npz"], allow_pickle=False) as t:
    before, after = t["img_before"], t["img_after"]  # [3,64,64,3] uint8 RGB
    frames = np.concatenate([t["frame0_img"][None], after])
    commands, proprio, jaw, valid = (t[k] for k in ("issued_mean","delta_eef","delta_jaw","valid"))
P_ref = np.asarray(row["views"]["nominal"]["P"])
P_observed = np.asarray(v["P"])
```

Each camera field is fixed across the four boundary images at controls 0, 5, 10, 15. `P` maps homogeneous world metres to pixel homogeneous coordinates. Divide by positive depth. The image coordinate order is column, row from top left, integer pixel centers, including (31.5,31.5) at image center. `K` is signed for recorded col direction. `world_to_cv` is a rigid transform into physical camera axes +X right, +Y down, +Z forward. `P = K @ world_to_cv[:3]`. `gl_to_world` stores the physical MuJoCo rotation, whose local +X is right, +Y is up, +Z is backward. `T_view_from_ref_cv` is observed-from-nominal SE(3), and `H_view_to_ref` is the inverse sampler direction. The reference plane is horizontal at `plane_z_m`, the public TCP z at reset. Equalize inputs across readers. Nominal pixels on a held-out moved-view trial serve only to score agreement.

Positive orbit is a right-handed world +Z rotation of center and orientation about the initial TCP. Positive local optical roll is right-handed about the camera's +Z backward axis, center fixed. Neither sign convention specifies a displayed raster operation. The historical framebuffer is 256 square, in OpenGL bottom-up order; native processing reverses rows and columns and area-pools 4×4 cells. The released 64×64 arrays are actual sensor-derived RGB at the native reader resolution. They have a mirrored horizontal axis versus the usual top-down pinhole raster. Calibration uses focal magnitude from the compiled model's vertical fov; negative recorded horizontal focal length implements that mirror.

In the measured release each view also has `motion_npz`, containing `mp` float32 [3,64,64,9] and `dis_flow` float32 [3,64,64,2]. MP calls the repository's DIS implementation followed by the released dis_lumaflow feature constructor on these exact raw view-specific images. To canonically resample RGB, recompute motion features from the resampled before/after pairs. The diagnostics file contains independent rendered landmark centroids, world TCP endpoints and the hidden physical commands. The scoring program is its consumer. Each raw tape carries legacy `delta_joint` for provenance, which the public loader omits.
