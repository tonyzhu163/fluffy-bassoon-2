# Native Goal 5 source-training layouts

R2 has 28 released training layout IDs. This augmentation draws new placements from the published **BASE** BDDL for “push the plate to the front of the stove”. A layout consists of four movable objects and three fixtures. Panda and Sawyer share one training group for the same draw. The number of admitted layouts, the number with a successful tape per arm, and the number of positive tapes are separate measurements.

## Camera-free smoke

Use the qualified R2 simulator interpreter on Linux, with the R2 and both sibling code directories in their existing relative locations. From the R2 code directory, run the following command with a fresh destination.

```sh
"$CROSSARM_SIM_PYTHON" -Bu generated_layouts.py \
  --seed-start 283100 --count 4 --max-seeds 4 --out /path/to/fresh/smoke
```

This tries seeds 283100–283103. It prints the world XYZ of the bowl, cream cheese, plate and wine bottle in metres. A pass requires four collision-admitted draws with four distinct configurations of movable XY positions. The index gives the group, native seed, state fingerprint, scene hash and positions of each draw. Its adjacent `.npy` contains the exact pre-settle Panda state. A rejection records the seed and reason. The command exits nonzero if fewer than four qualify. A repeated XY configuration counts as one layout even if other simulator coordinates differ.

## Collection

In the qualified R2 GPU environment, the same seed reproduces the smoke draw. The collector first runs camera-free searches using the R2 paddle and sweep candidate order, starting with the proven rim push. It sends the first successful command sequence to a fresh normal-camera reset. A failed replay can advance to the next candidate. Search stops at the first accepted tape per arm and layout or at the route budget.

```sh
"$CROSSARM_SIM_PYTHON" -Bu generated_collect.py \
  --seed-start 283100 --layouts 4 --max-seeds 4 --candidates 8 \
  --bodies Panda Sawyer --out /path/to/fresh/camera-bank
```

For **128 distinct training layouts**, the sampler alone accepts `--count 128 --max-seeds 160`. For source collection, the corresponding command is:

```sh
"$CROSSARM_SIM_PYTHON" -Bu generated_collect.py \
  --seed-start 284000 --layouts 128 --max-seeds 160 --candidates 8 \
  --bodies Panda Sawyer --out /path/to/fresh/bank128
```

The latter admits at most 128 draws from 160 consecutive seeds, tries at most eight routes per arm per admitted draw, and renders at most one replay per positive search. To distribute work, give processes separate fresh output directories and nonoverlapping **seed budgets**. The number of successful tapes can differ between arms. Exit code 2 means the collection fell short of the requested layout count or at least one requested arm had zero camera positives. Inspect `admitted_layouts`, `accepted_by_body` and `paired_positive_groups` for paired coverage. The index retains failed search and replay outcomes.

## Inputs and native reset

| Input | Identity |
| --- | --- |
| Native simulator | LIBERO-Pro2.0 `e5500c690c1498eca1b7050350654964190cdf70`, robosuite 1.4.0, MuJoCo 3.2.3 |
| Task | `libero_goal`, task 5, `BASE`; manifest SHA-256 `71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba` |
| Placement and success BDDL | `assets/bddl/libero-pro2.0-1680-v1/libero_goal/00_base/BASE__push_the_plate_to_the_front_of_the_stove.bddl`, SHA-256 `c2db20071f58fd43fe029e5bf9176f3c7371ee62cadc311f2b8682c089174c0d` |
| Public exact-duplicate reference | `init_files/libero_goal/push_the_plate_to_the_front_of_the_stove.pruned_init` in the pinned simulator source, SHA-256 `794beed679b18bbd7d9fb851c81877a974915ad63addff2d5ac5af0607fafac9` |

The R2 Linux launchers set `CROSSARM_SIM_PYTHON` to `$setup/env/bin/python`, `CROSSARM_RELEASE` to the pinned release, `LIBERO_PRO_ROOT` to `$setup/simulator-source`, and `LIBERO_CONFIG_PATH` to `$setup/libero-config`. The configured `init_states` directory must contain the pinned public bank. Put the release, simulator source, R2 code and sibling PaliGemma R1 directory on `PYTHONPATH`. Published HDF5 episodes and policy weights are unnecessary for these CLIs. Render replay requires R2's allocated NVIDIA EGL renderer. Its vendor and renderer strings are stored with the tape receipt. [CPU environment](cpu_smoke.sh) · [R2 GPU environment](run.sh)

For each uint32 seed the camera-free Panda wrapper calls `ControlEnv.seed(seed)`, `set_perturbation_seed(seed)`, and `reset()`. The pinned domain seeds NumPy's placement RNG. Its ordinary non-deterministic-reset path invokes the published placement initializers. It can retry placement failures under the same seed. `get_sim_state()` supplies the pre-settle Panda float64 row containing time in seconds, generalized positions, then generalized velocities in their native joint units. The row fingerprint uses the release's `numerical_state_fingerprint`, including its dtype and shape descriptor. Each generated row receives an exact membership check against all 50 public hashes. The public inputs contribute fingerprints only to the admission decision. No public rollout, image, action or reward enters route search. [Native wrapper](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/env_wrapper.py) · [Native BDDL sampler](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/bddl_base_domain.py)

The flat row omits the cabinet, stove and wine-rack model-level poses. The receipt saves their positions and quaternions alongside named non-arm joint positions and velocities. `scene_hash` covers this complete pre-settle scene using the release's canonical JSON hash. A separate `movable_xy_hash` measures diversity in the four movable centres. At admission, scratch MuJoCo data recomputes contacts at the recorded coordinates. The seven task entities must satisfy the native deep-contact check (2 mm). Contact at nonpositive distance across a task entity's boundary is also rejected except for contact with the table support. MuJoCo signed geometry-distance queries also check distinct welded fixture bases, which the constraint solver filters. A start already satisfying the BDDL goal is rejected.

For collection, R2 opens a fresh session and the generated reset supplies its private singleton Panda row. The sampler seed regenerates model-level fixture transforms. On Sawyer, R2 copies the named scene and fingers from the native Panda donor and solves its stock bounded grip-site IK. Every actual pre-settle scene hash must match the sampled receipt. R2 applies ten native settling controls, checks the paired task-body poses within 1 mm and 0.01 rad, and begins the BASE rollout clock. The generated collector also checks placement contacts at the settled start. Geometry and collisions are read to generate demonstrations. All actions use the stock PandaGripper and 20 Hz relative OSC. [Reset adapter](generated_layouts.py) · [R2 arm reset](../libero-crossarm-20261120-r1/arm_env.py) · [Matched scene transfer](../libero-crossarm-20261120-r1/sim_reset.py)

## Tapes and provenance

The camera rollout starts at control zero and uses the saved search commands. Admission requires the same settled state fingerprint, the same control count, native termination and native success, zero prior rewards, and terminal reward one. The manifest horizon is 300 controls (15 s). `--max-steps` can shorten a search. The two 256 × 256 uint8 RGB views are `agentview` and `robot0_eye_in_hand`. Each tape retains the original `save_tape` arrays, including pre-action state, issued controls, measured motion, rewards, and terminal state. Observed state has TCP XYZ in metres, hand quaternion XYZW and signed jaw opening in metres. Actions have seven normalized components in [−1, 1], with per-axis translational scale 0.05 m, rotational scale 0.5 rad and a signed gripper command. Motion measures translation in metres, rotation vector in radians and jaw displacement in metres. [R2 expert](paddle.py) · [Route search](sweep_bank.py) · [Tape writer](../libero-paligemma-crossarm-20261120-r1/demos.py)

The index, episode row, NPZ `pose`, and reset `init_state_id` all use the training group **1,000,000 + layout_seed**. `bank_row_index=0` records the private row slot. `layout_origin` in the row and tape binds this group to the native API, actual sampler seed, source kind `native_bddl_sample`, collision check, row fingerprint and scene hash. The reset uses `generated_row_fingerprint` for the raw Panda state and `initial_state_fingerprint` for the settled state of the executing arm. The run index records the code revision and SHA-256 of the implementation files. The layout index also records the SHA-256 of the serialized `.npy` file. This file digest and the release's numerical fingerprint have different byte encodings.

## Downstream source-only integration

Dispatch a bank explicitly marked `source_kind=native_bddl_sample` to the generated helper. The existing motor loader rejects these groups until wired. Use the original tape, reward, renderer, file checksum and deduplication checks together with the following calls:

```python
from generated_split import validate_generated_attempt, validate_generated_groups, validate_generated_tape, validate_layout_file
from pretrain_reader import NATIVE_RESET

for layout in index["layouts"]:
    validate_layout_file(bank_root, layout)

# For each accepted row of a generated index, with its pickle-free NPZ open:
fields = validate_generated_attempt(index, row, native_reset=NATIVE_RESET)
validate_generated_tape(tape, row)
record.update(fields)

# At source fit and saved-fit readback, check all generated records together.
validate_generated_groups(generated_records, index["public_row_fingerprints"], native_reset=NATIVE_RESET)
```

Use the returned `pose` to group trials and `split="train"` in source sampling and partitioning. For generated coverage, count `generated_row_fingerprint` and unique training groups per arm. The helper verifies the ordered denylist against a pinned digest and rejects conflicting group origins or duplicate physical layouts across groups. Keep the released branch's existing numerical split and confirmation panel. At transfer fit, dispatch generated records only for source training, retaining the per-body reward, camera and source-validation checks. [Source-only helper](generated_split.py) · [Motor loader](motor_bank.py) · [Transfer fit](transfer_fit.py)

Implementation and static review used repository base `7e9e8ffce83bd584dc3e8ef3cafcbe8480a7c300`. Native seed diversity, Sawyer admission rate, renderer replay success and camera tape counts await execution on the qualified Linux runtime.
