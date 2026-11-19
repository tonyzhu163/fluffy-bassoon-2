# Motor pilot interface

Run `gate.py --output DIR` with the qualified JAX OpenPI interpreter. It creates a fresh `DIR/pilot/` for the 96 episodes. Adding `--smoke` creates `DIR/smoke/` for one full native episode on pose 30 with reversed X, D17, the correct decoder and regrasp enabled. Each mode refuses to overwrite its own directory. Both use one visible CUDA device and the qualified EGL simulator interpreter.

Within the mode directory, `episodes.jsonl` appends one JSON object per completed episode after its arrays are saved. The primary fields are `pose`, `wire` (+1 nominal, −1 reversed), `condition` (BASE or D17), `decode` (+1 identity, −1 invert X), `regrasp` (0 or 1), `native_success` (boolean), `seed`, `initial_physics`, `initial_public`, `first_proposal`, `d17_opportunity`, `d17_triggered`, `d17_event_control` and `controls`. `complete.json` appears only after every planned episode passes the runtime checks. Analyze the full pilot from `DIR/pilot/episodes.jsonl` after checking its completion marker.

## Controls and pairing

Each branch creates the released BASE or D17 environment on the specified pose and seed. D17 is rearmed at every D17 reset. The native adapter performs its standard ten settling steps before control zero. Goal 2 permits 300 subsequent controls, with immediate stopping on native termination or truncation. The policy receives the released BASE instruction. Its two 224-pixel cameras and proprioception come from the current branch. Each ordinary inference executes the first five controls or stops early on native completion.

The regrasp-enabled branch latches the twelfth consecutive positive gripper command and starts the maneuver at the next five-control boundary. For 30 controls the jaws open while the wrist servos toward its intervention-start pose plus 0.10 m vertically. For 30 more open controls it servos toward the public wrist pose eight controls before the first closure in that streak, clamped to reset. Each servo step uses the latest wrist pose. A fresh motor query follows the maneuver if the episode continues. Both X transforms apply to every commanded action, including the servo.

Both reset and flow-noise seeds are 19193000 + pose. A NumPy default generator draws the 60 physical-time slots of 10 × 32 standard normals in order and casts them to float32. OpenPI receives the noise for the current slot and its JAX key folded with the same slot. A maneuver skips its slots. All 16 branches per pose must match on reset physics, public observables, prompt, action schema, noise schedule and opening motor input and proposal. Each of the 48 equivalent net-sign pairs must also match on the full saved physical trajectory and live queries. The two members are executed independently.

## Episode evidence

Each JSONL row names its episode directory. Array files use NumPy NPZ with numeric or boolean arrays and support `allow_pickle=False`.

| File | Contents |
| --- | --- |
| `reset.npz` | Original public camera images and all public robot fields at control zero. |
| `public.npz` | `robot_state[0:T+1]` holds wrist XYZ, quaternion XYZW and two jaw coordinates. `phase` is 0 for the motor, 1 for retreat and 2 for return. `selected` is the pre-clip controller output. `wire_action` follows clipping and the chosen X decoder. `reward`, `success`, `terminated` and `truncated` correspond to each executed control. |
| `public.npz` query fields | `query_control` counts completed controls before inference. `query_rgb` holds both motor-facing cameras (agent, wrist). `query_state`, `query_noise`, `query_jax_key` and the entire `query_proposal` (10 × 7) share its query index. `final_rgb` and `final_state` are the terminal policy-facing observation. |
| `evaluator.npz` | `native_input` is the action delivered after the hidden X convention. `applied` is the release's actual post-perturbation command, before OSC physics. BASE must preserve every dimension. D17 may force the gripper open. |
| `diagnostics.json` | Release diagnostic read after termination. D17 opportunity counts the logged grasp-failure window start. Activation counts the released recoverable-grasp-failure event. Raw event steps are zero-based and `d17_event_control` is one-based. The public closure latch is recorded separately. |
| `result.json` | Episode row plus per-array trajectory fingerprints. `design.json` at the mode root records the schedule, source revision, source hashes, checkpoint identity and fixed controller settings. |

Successes establish which motor and recovery choices are attainable for each assigned condition. The comparison uses oracle-selected corrections to measure headroom. A history benefit requires a separate support/query experiment.

## Static verification

On Mac arm64, both files compiled and parsed with the Python 3.11 grammar. Source inspection checked 96 unique pilot keys, 48 equivalent-sign pairs, the one-episode smoke key, simulator RPC names, scorer row fields, and both task BDDL hashes against the pinned manifest at `e5500c69`. The SHA-256 digests were `8152afad70a4253e9c653f1800aa2066d1e977c69ffb03d035a30b39469ec5ea` for `gate.py` and `4e16d0a638da9f9263dad923d49a4a4b37225c7c0a10e54e1ba8eca02dec8a67` for `grasp_simulator.py`. Live GPU execution and recovery outcomes remain unverified by this worker.
