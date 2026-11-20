# Cross-arm simulation interface

Use `arm_env.ArmEnv` inside the pinned LIBERO simulator process. The case uses a released BASE task and initialization. Panda and Sawyer are the source bodies. UR5e is the held-out body. All three stock arm models and wrist cameras are present in the 1.4.0 wheel. Each arm uses the same physical PandaGripper.

```python
from arm_env import ArmEnv

with ArmEnv(arm="Sawyer", suite="libero_goal", task_id=1) as env:
    first = env.reset(init_state_id=0, seed=17)
    model = first["model_identity"]  # audit only
    raw, prompt = first["observation"], first["language_instruction"]
    result = env.step(action7)
    # probe(action7) also executes exactly one live step and returns the same fields.
```

`reset` returns `observation`, `language_instruction`, `model_identity`, `environment_reproducibility`, `max_steps`, and `worker_step_index=0`. `step` and `probe` return `observation`, `language_instruction`, `issued_action`, `sensor_motion`, native scalar `reward`, booleans `success`, `terminated`, `truncated`, and `worker_step_index`. Both consume a finite bounded 7-vector in [-1, 1] and advance the same 20 Hz episode clock. Motion is the observed one-control change in world TCP XYZ (metres), the world-frame relative rotation vector (radians), and the change in jaw opening (metres, positive opens). Each is a float32 7-vector. Run probes in a separate reset for separate support episodes. The caller owns subprocess isolation and policy inference.

`observation` contains the seven standard Pro2 raw fields including two 256×256 uint8 views, native-length arm joint position/velocity, world grip-site position, world hand-link quaternion XYZW and two common finger slide positions. Pass raw observations and the instruction to the prior run's `motor_iface.to_openpi_observation`. This yields the two oriented 224×224 views and 8D policy state. Issue its decoded 7D OSC plus gripper proposal, bounded by the caller. Keep model identity and native-length joint arrays out of learned context. The model identity is also available as `env.model_identity` after reset. BASE is the sole variant for this physical-body study.

Source and qualification details will follow implementation.

The cross-body pairing key is `environment_reproducibility["scene_fingerprint"]`, computed before settling over the released named task joints/velocities and fixture poses read back from the target. The within-body key is `environment_reproducibility["initial_state_fingerprint"]`, the released hash of the full settled MuJoCo state. `model_identity["arm"]` is the canonical case-sensitive family name; `joint_count` is 7, 7 or 6 respectively. Compare post-settle placements separately during qualification.
