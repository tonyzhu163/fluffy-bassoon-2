# Stage 2C CPU development qualification

Use the released training task `libero_goal/1`, “put the bowl on the stove,” which the project reserved for development. Use published state 13 and seed 20500101 throughout. One support executes the existing 15 prescribed OSC controls. Its independent query stops at frame zero. The question is whether a stock arm admits the published start and whether a released visual case produces its specified effect under that arm.

**Physical identities.** Use release `e5500c690c1498eca1b7050350654964190cdf70`, the hydrated simulator, MuJoCo 3.2.3 and robosuite 1.4.0 from wheel SHA-256 `aba065e7b36745738cede259457b2cb349427f3608728d867ef3a2034cb62994`. The compiled tabletop models must be MountedUR5e (six joints), MountedIIWA (seven joint_1…7 hinges) and MountedKinova3 (seven Actuator1…7 hinges). The arm and common PandaGripper MJCF hashes are pinned in arm_env.py. Each uses its own arm geometry, placement registration and OSC controller. Kinova3 has continuous hinges at 1, 3, 5 and 7. BDDL hashes, operators and targets for all six cases are fixed in qualify.py.

**Admission.** First run BASE on each arm. Require the named donor-scene, bounded IK and settled task-body checks in ArmEnv. Record donor-to-arm settled tool errors. Require both 256² RGB views and exact independent support/query reset records. The probe must finish all 15 controls, move the TCP more than 15 mm across its three bursts, move the jaw more than 10 mm in one burst, and move the external and wrist images (mean burst L1 above 0.5 and zero, respectively). These use the prior development excitation check and add wrist motion readback. A failed arm BASE blocks its case comparisons.

**Paired cases.** Compare UR5e BASE with S13, S14, S15, S16 and S19, then test S15 first and the four remaining cases on the first passing arm in IIWA, Kinova3 order. Test S15 on the second arm too if both pass. Require identical released row, pre/post-settle scene, full settled state, TCP, jaw and prompt within each arm. Collect support anew for each case and preserve both RGB streams at burst boundaries. Check distinct effects against that arm's BASE frame zero.

| Case | Required observation on Goal 1 |
| --- | --- |
| S13 | Bowl and stove geometry receive RGB (0.18, 0.72, 0.38). Other geometry colors match BASE. At least one view changes. |
| S14 | Visible geometry follows the published palette mix (0.48). Ambient and diffuse lighting scale once by 0.64. At least one view changes. |
| S15 | Both cameras translate by (0.10, −0.10, 0.07) m in their parent frames with unchanged quaternion. Both views change. |
| S16 | Both vertical FOVs scale by 1.35 with release clipping. On both cached source RGB frames the released shift, crop and distortion reproduce the public pixels exactly and visibly change them. |
| S19 | Both pre-transform RGB frames equal BASE. Applying intensity ×0.82 and seed-matched Gaussian noise (SD 12) reproduces both published RGB frames exactly. |

The Linux driver writes results.json after each cell. Each cell records native rewards, issued controls, tool motion, plan, artifact hashes and failed predicates or exceptions. A native crash or time limit is recorded by the parent driver. The sequential panel has a 450-second wall-clock budget. Exhaustion leaves explicit skipped rows. The output is a simulator qualification on one start. Policy task success requires a separate experiment.

## Linux invocation

From the code checkout, on the existing Baige development CPU host, run the following with a fresh output path.

```bash
bash outputs/xemb-stage2c-cpu-20261121-r1/run.sh
```

The default receipt is `/mowenjie/fluffy-bassoon-2/outputs/xemb-stage2c-cpu-20261121-r1/cpu01/results.json`. Pass a fresh directory as the first argument for another attempt. The wrapper selects the existing uv-managed Python 3.11 simulator environment in `/mowenjie/fluffy-bassoon-2/setups/libero-meta-learning-20261117-r1`, including NumPy 1.26.4, SciPy, Torch for released state-bank loading, MuJoCo, robosuite, PyOpenGL, LIBERO config and hydrated assets. It selects Mesa software EGL device 0 and hides accelerators. Set `XEMB_SIM_SETUP`, `CROSSARM_RELEASE` or `XEMB_STORAGE` to relocate the corresponding prepared directories. Requires Git, the pinned release checkout and Mesa's GLVND vendor file at `/usr/share/glvnd/egl_vendor.d/50_mesa.json`.
