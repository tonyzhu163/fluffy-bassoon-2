# World-frame control on the archived UR5e supports

**The original 96 passive tapes support an independent world-frame measurement.** Each stores the measured changes in all six physical joint angles at three successive chunk boundaries. From the matching reset, accumulate those changes and evaluate the pinned UR5e and PandaGripper forward kinematics (FK) at the world grip site. Pair the resulting displacements with the clipped, actually issued chunk means from the same tape. The original passive dynamics and individual primitive commands need no reconstruction for this estimate. The parent’s Baige Linux CPU run identified yaw on all 96 tapes with 1.93° mean absolute error (MAE) overall and 2.02° on the 64 OOD tapes.

## Verified original-trace result

| Panel | Identified / total | Wrapped yaw MAE |
|---|---:|---:|
| All original tapes | 96 / 96 | 1.93° |
| Familiar | 32 / 32 | 1.76° |
| Compositional OOD | 32 / 32 | 2.21° |
| Continuous-angle OOD | 32 / 32 | 1.82° |
| Both OOD regimes | 64 / 64 | 2.02° |

The minimum measured three-chunk XY path was 53.78 mm. All eight nominal active qualifications reproduced every archived joint and public world TCP delta exactly at float32 precision. All eight whole-simulator reset hashes differed. These are numerically qualified robot resets. The underlying state differences responsible for the hash mismatches are unknown. Across 288 passive chunk displacements the FK versus archived planar-length discrepancy was 0.078 mm median and 0.945 mm maximum. The corresponding vertical discrepancies were 0.022 mm and 0.160 mm. The full XYZ discrepancy on the 48 zero-yaw passive chunks was 0.112 mm median and 0.449 mm maximum.

The parent ran `world01` at Git revision `a57df8c7672eb837cf53f314fc1c4f9cdd1bff3f` on Baige CPU (128 seconds, exit 0, 0 GPU-hours). Full JSON on shared storage is `/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-command-20261123-r1/world01/world.json`, SHA-256 `f6b5dcbcb9f79cd57210b23239c898a46926a0f8783e60cd1db6177728fef3b6`. This is a yaw-estimation measurement on the original trajectories, with eight tasks providing the independent task clusters.

## Archive audit

Read-only inspection of all eight Baige shards found 96 passive NPZs, 96 active siblings and eight support manifests in `gpu09test/test_live`. The passive file-set SHA-256 is `8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554`, hashing each file's lowercase hex digest plus newline in shard and slot order. All 96 passive tapes have three valid slots, continuous before/after image pairs and one common initial state and frame-zero image per shard. Each manifest has four keys, `embodiment`, `physics`, `active_npz` and `passive_npz`. The original launch identifies commit `5dd5fa914bbbfe25b9cf4ca5fa504c8a472e997b` and job `job-n5lff6bmuwfc`. All eight logs report the query-stage `chunk_steps` exception after the supports were written. The logs contain no passive action vectors. [Official collector](../libero-xemb-ur5e-scale-20261120-r1/eval_test_official.py) · [Support provenance](RUNTIME.md)

| Stored field | Shape | Measurement |
|---|---|---|
| `delta_joint`, `joint_count` | 3 × 6, scalar 6 | Differences of independently observed native arm hinge coordinates, in radians and unrotated by the interface |
| `issued_mean` | 3 × 7 | Average of the bounded pre-interface controls the simulator actually executed in each chunk |
| `delta_eef`, `delta_jaw` | 3 × 3, 3 × 2 | Command-frame EEF displacement, change in finger opening and final opening |
| `frame0_state` | 8 | Reset EEF XYZ, hand axis-angle and two finger coordinates |
| `img_before`, `img_after`, `frame0_img`, `valid` | 3 × 64 × 64 × RGB each, one RGB frame, three Booleans | Upright pooled agentview at the chunk boundaries and the valid-chunk mask |

**Exact full-scene passive replay is blocked.** The data retain a reset selector and fingerprint rather than a serialized support reset, and omit per-primitive commands, per-boundary scene states, terminal flags and per-chunk executed counts. Averaging seven command channels loses the temporal order and clipping/contact history. The collector capped support at 15 primitive controls, five per chunk. Three valid chunks establish endpoints at the first two five-control boundaries and the end of the third chunk. An early termination within the last chunk cannot be independently excluded using the mask alone. The original evaluation treats these as 15-control tapes. Privileged query records came from layout 13, while these supports used layout 11. [Motor and tape construction](../libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py) · [Public observations and command interface](../libero-xemb-ur5e-sysid-20261120-r1/ur5e_sim_worker.py)

## Run the original-trace world control

Run in the parent-deployed checkout and pass a fresh result path. The program selects the collecting revision's simulator helpers and verifies their bytes. It uses the existing Linux simulator Python and software EGL. The default covers all eight shards. Add `--shards 0` for a smoke run with one task and twelve estimates.

```sh
root=/mowenjie/fluffy-bassoon-2
run=outputs/symmeta-camera-command-20261123-r1
py="$root/setups/libero-meta-learning-20261117-r1/env/bin/python"
env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 \
    "$py" -Bu "$run/world_control.py" --output "$root/$run/cpu01/world.json"
```

The control uses a paired `fam_nom` active sibling **only** to qualify each reset. It reissues the sibling's known fifteen native controls, compares all 18 joint differences and nine world TCP differences with the stored active measurements, and records the complete initial-state hash match. If hosts differ below float32 sensor precision, the report distinguishes numerical agreement from a bitwise physics match. Qualification requires the eight reset motor coordinates to agree within 2 × 10⁻⁶, the active joint differences within 2 × 10⁻⁶ rad and the native TCP differences within 10⁻⁶ m per component. The nominal active numbers never enter any passive fit.

For each passive tape, the script checks the fixed grip site's kinematic chain, reads the reset joint sensor and sums the three archived six-joint differences in float64. MuJoCo computes the site XYZ at each of the four measured arm configurations on scratch data. The estimator takes the angle of the summed cross and dot products of issued XY and measured world XY, with equal weights across chunks and no fitted intercept. It reads the yaw label only to score the estimate. Local EEF is used afterward to compare yaw-invariant XY lengths and vertical components. The two zero-yaw interfaces additionally permit a full native XYZ check. Robosuite samples joint encoders and cached Cartesian site positions on its observable clock. Their small timing differences and float32 quantization can produce a residual between joint FK and the public Cartesian sensor. [Pinned arm interface and FK](../libero-crossarm-20261120-r1/SIM.md) · [Implementation](world_control.py)

Join the JSON `rows` by `index`, `embodiment` and `archive_sha256`. Each includes the four `world_xyz_m` endpoints, three `delta_world_m` and `issued_mean` vectors, `yaw_hat_deg`, excitation and sensor consistency checks. The world measurement is shared across optical rolls of the same tape. `summary` reports wrapped MAE and the identified denominator overall, on OOD, by regime and by task. Inspect `qualifications` and require `status == "complete"`. No GPU or VLA inference is needed.

## Orbital render scope

The archive identifies the arm endpoints, but it does not identify the complete scene for exact orbital rerenders. A separate physical replay can hold each archived seven-channel command mean constant for five controls and record a fresh world TCP and nominal/azimuth views together. Any such result measures fresh held-mean trajectories. It must be evaluated against its own fresh zero-azimuth baseline.

The optional CPU worker implements that replay. Its defaults cover all twelve transforms of the first task at 0°, −30° and +30°. Run a single oblique case first or pass more shard/slot numbers. The output argument here is a fresh directory. The camera orbits about the world vertical through the reset grip site. At every boundary the worker moves the camera center and orientation together, preserving focal length. It restores the camera before integration resumes, checks identical MuJoCo integration state before and after rendering, and requires the nominal image and world grip-site reading at that sensor instant to match the native public observation. The complete scene is common to every view of a fresh state. [Orbital worker](replay_worker.py)

```sh
fresh="$root/$run/orbit-smoke01"
env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 \
    "$py" -Bu "$run/replay_worker.py" --shards 0 --slots 8 --output "$fresh"
```

`replay.json` contains every issued and physical primitive, native rewards, camera poses and measured world-frame fit. It records the per-boundary distance to the archived joint-FK positions, deviations from archived joint differences, and nominal RGB differences. Each angle has its own score-compatible NPZs and JSON descriptor with the same case IDs and index. Once `replay.json` has `status == "complete"`, use the separate reader environment to score each descriptor with optical roll fixed at zero. Compare across angles by the shared case ID. [Reader runtime](SCORING.md)

```sh
code=$(git rev-parse --show-toplevel)
reader=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
overlay="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay"
for angle in +0 -30 +30; do
    env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        PYTHONPATH="$overlay:$code/src" "$reader" -Bu "$run/score.py" --rolls 0 \
        --descriptor "$fresh/azimuth$angle.json" --output "$fresh/score$angle.json"
done
```
