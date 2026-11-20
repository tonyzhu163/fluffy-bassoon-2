# Native Goal 5 source teachers

The Sawyer hypothesis is that a tilted, closed gripper can get purchase on the *outside* of the plate by descending beside a flank and sweeping around to the trailing quadrant. The previous tilted inner-rim attempt on layout 0 recorded one finger–plate contact in 300 controls. It spent 219 controls in engagement, translated the plate 6.92 mm and earned zero native reward. An earlier descent on the same layout reported contact between link 4 and the wooden cabinet. Those results motivate changing both the approach side and the contact surface. Success of this new controller awaits native qualification. [Previous native receipt](../libero-paligemma-crossarm-20261120-r1/evidence/demoqual08/demos/index.json) · [Earlier contact readback](../libero-paligemma-crossarm-20261120-r1/evidence/demoqual06/run.log)

## First Sawyer search

Run these from the repository root in the orchestrator's qualified Baige environment. The existing launchers take the verified commit in `PIN` and create a fresh run directory under the shared run root. Use a new attempt name for each invocation. The following short search uses all six candidates on the diagnosed training layout. It writes its bank in the launcher's working directory. It requires the simulator interpreter and produces traces without rendering either camera.

```bash
here=outputs/libero-paligemma-dualarm-20261120-r2
shared=/mowenjie/fluffy-bassoon-2/outputs/libero-paligemma-dualarm-20261120-r2
bash "$here/cpu_smoke.sh" "${PIN:?}" edgequal01 expert.py search \
  --layouts 0 --all-candidates --out search
```

Each candidate starts from its own normal reset on exactly the same layout and seed. The initial candidates vary side, trailing contact angle, tilt, yaw, pad height, push depth and whether to pass beside the cheese. `left-arc` and `right-arc` mirror the contact side at the same tilt and yaw. `rear-oblique` provides a nearly rearward push along a direct route. The controller recomputes the plate's outer support plane and the leading low finger surface from compiled collision vertices each control. The gripper targets a positive gap from the plate during descent, follows its exterior arc, then commands a small overlap and tangential lead as the plate travels. The detour uses the reset cheese footprint and plate radius. Each new leg lifts and repositions the fingers. [Controller and parameters](expert_sweep.py)

Read `search/index.json` for outcome, first plate contact, first TCP stall and best decrease in plate-to-goal distance. Each attempt links to a JSONL trace. The opening record contains the reset, initial scene, active contacts and plate waypoints. Each subsequent control records the issued action, pre-action public state, measured motion, post-action TCP, plate, finger height, actual contact pairs with normal force, commanded target, edge-gap error and recent motion. Measurements are sampled at 20 Hz control boundaries. The last record summarizes the attempt. Sawyer receipts include final joint-limit margins, finger bounds and the signed distance for the pilot’s link-4/cabinet pair, capped at 10 cm. A TCP stall records the plate and TCP positions and best goal progress before its 12-control window.

```bash
jq '.attempts[] | {route,steps,outcome,best_goal_gain_m,first_plate_contact,first_tcp_stall}' \
  "$shared/edgequal01/search/index.json"
jq -c 'select(.step != null) | [.step,.control.phase,.control.actual_edge_gap_m,
  .post_scene.finger_low,.goal_gain_m,.tcp_span_m,
  [.active_contacts[] | select(any(.geoms[]; test("robot0|gripper0"))) | .geoms]]' \
  "$shared/edgequal01/search/traces/Sawyer-0-r0-search-left-arc.jsonl"
```

If the fingers stop above the rim with a loaded arm–fixture pair, change the approach side or tilt. If they reach the rim and sustained contact produces little plate motion toward the goal, change the trailing angle, contact height or push depth. If the plate makes progress and stops beside the cheese, compare the direct route with the detour. The controller abandons a stationary waypoint and starts the next candidate on a fresh reset. Finger contact and plate movement serve as diagnostics. Native terminal success with positive reward is the collection criterion.

For a targeted parameter change, write a JSON list such as `[{"name":"left-low","flank":-75,"trailing":-20,"height":0.004,"tilt":48,"yaw":-90}]` on the shared volume, then add `--params /absolute/path/candidates.json --candidates left-arc,left-low`. Omitted fields take their defaults from `Candidate`. `--all-candidates` keeps testing after a search success. Search accepts training layouts 0–27.

## Render a result and collect a bank

This command reissues the successful search controls in an ordinary camera-enabled Sawyer session from the same layout and seed. A search result by itself supplies no training tape.

```bash
bash "$here/run.sh" "${PIN:?}" edgerender01 sim expert.py collect \
  --bodies Sawyer --layouts 0 --search-bank "$shared/edgequal01/search" --out bank
```

For larger banks the same command can run Panda official replay and Sawyer search concurrently in two processes in the allocated container. The launcher's `DUAL_ARCHIVE` supplies the pinned Goal 5 BASE HDF5 file. These examples request four episodes on each training layout and one on each offline development layout. Use a fixed trained-on-source candidate order for offline collection. Pass an explicit comma-separated list after `--candidates` to use a previously chosen order.

```bash
bash "$here/run.sh" "${PIN:?}" edgefit01 sim expert.py collect \
  --bodies Panda Sawyer --layouts 0-27 --repeats 4 --search-first --out bank
bash "$here/run.sh" "${PIN:?}" edgeoffline01 sim expert.py collect \
  --bodies Panda Sawyer --layouts 28-31 --repeats 1 --out bank
```

`--search-first` tries each Sawyer candidate with cameras disabled and renders a positive command sequence on an independent normal reset. Without this option the native Sawyer controller runs directly with cameras. The camera-free lane calls the original arm reset and step with the release's `ControlEnv` camera observations and offscreen renderer disabled and proprioception normalized using an empty camera list. It retains the compiled robot and camera geometry, released layout transfer and settle, 20 Hz OSC, horizon and reward. The rendered replay records whether the search and replay initial-state fingerprints match. Its own native outcome determines acceptance. Each attempt writes its receipt atomically into the same bank index. `quality.json` reports successful requests, distinct layout IDs and distinct initial-state fingerprints. An incomplete collection exits with status 1 and retains all accepted tapes and failure traces.

Panda tries each matching official trajectory in the pinned archive at its recorded seed on the first pass. It issues the released 7D commands and records the live pre-action observations. With no matching success, it can track a natively replayed donor using the pilot's measured wrist and plate feedback. The receipt identifies the route, donor layout, demo ID and seed. `--official-only` restricts Panda to unmodified matching commands. Defaults for fallback donors are training demo IDs 22, 1, 0 and 6. A matching offline-dev demo is used only on its own layout. The collector rejects requests for layouts 32 and above. Fresh online source dev 32–35 and final 40–47 belong to the evaluation driver. [Official replay implementation and archive identity](../libero-paligemma-crossarm-20261120-r1/oracle_fallback.py)

Requested target seeds are `seed_base + layout + 1000 * repeat` with base 20261120 by default. The first matching Panda replay uses its published seed. The Sawyer candidate order rotates between repeats. Count repeated seeds, donor trajectories and target layouts separately when reporting coverage. The old loader hard-coded poses 0–15. This run's motor loader reads the expanded source split and repeated trajectories from the bank index. [Motor input contract](motor_bank.py)

## NPZ contract

One accepted episode has path `BODY/POSE.npz` for repeat zero or `BODY/POSE-rREPEAT-sSEED.npz` for later repeats. The writer is the same one used by the previous motor. Every accepted row of `index.json` gives the relative `file`, `body`, `pose`, `seed`, `steps`, `done`, `native_success`, `terminal_reward` and the native `reset` receipt. Privileged coordinates and contact diagnostics live in the JSONL traces. [Native tape writer](../libero-paligemma-crossarm-20261120-r1/demos.py)

| NPZ key | Type and shape | Timing and content |
| --- | --- | --- |
| `rgb`, `wrist` | uint8 [T,256,256,3] each | Native agentview and eye-in-hand frames immediately before each action |
| `state` | float32 [T,8] | Pre-action public XYZ, hand quaternion XYZW and signed jaw opening |
| `actions` | float32 [T,7] | Actual bounded 20 Hz relative OSC XYZ, rotation and gripper command issued at that frame |
| `motion` | float32 [T,7] | Measured public motion resulting from the corresponding command |
| `rewards`, `terminal_state` | float32 [T], float32 [8] | Sparse native rewards and final post-action public state |
| `prompt`, `body`, `pose`, `seed`, `native_success` | Unicode, Unicode, integer, integer, bool scalars | Native instruction, source arm, released row, actual seed and success receipt |

Both Python files compiled on macOS. Static inspection resolved the imported names against the sibling simulator, replay and tape-writer modules. All four launcher examples passed shell syntax and CLI parsing. Argument checks rejected search on 28–31, collection on 32–35 and 40–47, duplicate layouts and zero repeats. Native search, rendered replay parity and successful NPZ production still require Baige qualification.
