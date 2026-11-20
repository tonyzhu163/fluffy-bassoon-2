# Source-supervised PaliGemma waypoints

Can a camera-localized plate and a continuous wrist-motion target improve fresh Goal 5 pushing on both source arms? Test the same checkpoint on the four registered development layouts per arm. Compare its native successes with the existing raw-action fit on the same starts and, for an architecture comparison, the same accepted camera tapes. A lower offline action error alone does not establish better pushing. [Source protocol](PLAN.md)

## Commands for the allocated Linux job

Run each launcher command from the pinned repository root in a Baige Distributed Training allocation. Set `PANDA_BANK` and `SAWYER_BANK` to existing native Paddle camera bank directories with an `index.json`. Extra banks can follow `--banks`. The first command snapshots the attempt lists at invocation and labels every accepted Paddle trajectory on layouts 0–31. The launcher selects the simulator or VLM interpreter, renderer, offline base weights and code revision. Each launcher attempt name must be fresh.

```bash
set -eu
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
PIN=$(git rev-parse HEAD)
bash "$run/run.sh" "$PIN" wplabel01 sim waypoint_labels.py \
    --banks "${PANDA_BANK:?}" "${SAWYER_BANK:?}" \
    --workers 2 --camera-fallback commands-and-route --out bank
```

Inspect skips and native layout coverage before fitting. A completed labeler with zero accepted tapes exits 2. Infrastructure or data-integrity errors leave the bank incomplete. A fresh label attempt can use the same immutable inputs.

```bash
labels="$campaign/wplabel01/bank"
jq '{status, accepted, duplicate_input_tapes, sources}' "$labels/index.json"
jq -r '.attempts[] | select(.waypoint_provenance.skips | length > 0)
       | [.body,.pose,.accepted,.waypoint_provenance.mode,
          (.waypoint_provenance.skips | join(" | "))] | @tsv' "$labels/index.json"
jq '[.attempts[] | select(.accepted)] | group_by([.body,(.pose < 28)]) |
    map({arm:.[0].body, training:(.[0].pose < 28), tapes:length,
         layouts:(map(.pose) | unique)})' "$labels/index.json"
```

For a fit against the existing offline tapes, run the following two commands in successive allocations. Training requires at least two successful training layouts and one successful offline layout per arm. The launcher places the fit under the attempt directory and development writes to that same fit.

```bash
fit="$campaign/wpfit01/fit"
bash "$run/run.sh" "$PIN" wpfit01 vlm waypoint_train.py train \
    --demos "$labels" --output "$fit" --steps 800 --candidate-steps 200 400 800 \
    --shortlist 3 --microbatch 4 --anchors-per-pose 32 --seed 20261120
bash "$run/run.sh" "$PIN" wpdev01 vlm waypoint_train.py dev --output "$fit"
jq '{source_gate,best_dev_step,selected_step,motor_checkpoint_sha256,scores}' "$fit/dev/report.json"
```

The campaign also uses a fixed-teacher offline cohort. To construct that cohort, label the wider search banks with `--fixed-offline` and supply a separately fixed offline camera bank. The following alternative takes one prespecified original Paddle route per body/layout/seed and captures frames and labels together on the native camera step. It retains every rejection and native failure. It works even when camera-free replay cannot reproduce any trajectory. Use this version of the fit for a comparison requiring fixed offline labels.

```bash
bash "$run/run.sh" "$PIN" wptrainlabel01 sim waypoint_labels.py \
    --banks "$PANDA_BANK" "$SAWYER_BANK" --fixed-offline --workers 2 \
    --camera-fallback commands-and-route --out bank
bash "$run/run.sh" "$PIN" wpfixed01 sim waypoint_capture.py \
    --bodies Panda Sawyer --poses 28-31 --seed 20261120 --out bank
fixedfit="$campaign/wpfitfixed01/fit"
bash "$run/run.sh" "$PIN" wpfitfixed01 vlm waypoint_train.py train \
    --demos "$campaign/wptrainlabel01/bank" "$campaign/wpfixed01/bank" \
    --output "$fixedfit" --steps 800 --candidate-steps 200 400 800 --shortlist 3 --microbatch 4
bash "$run/run.sh" "$PIN" wpdevfixed01 vlm waypoint_train.py dev --output "$fixedfit"
```

The fixed-offline filter follows the campaign snapshot convention and drops layouts 28–31 from source indices bearing `source_constructor`. Record the collector provenance of every other supplied offline bank. The default all-tape command includes any offline demonstrations selected by search. Offline rankings from those rows describe that selected cohort. Preserve the full generated bank, including `aux/`, when copying it. The original raw-action snapshot utility copies only camera NPZs.

## What a label certifies

The replay uses the original native ArmEnv reset and OSC step. It reads full-precision plate world XYZ before issuing the stored control. For every transition it checks the native step index, exact float32 issued action, next measured motion, reward and done/success flags. It compares every pre-action public wrist and jaw and the final public state, including quaternion angle modulo sign. Both released and initial-state fingerprints must match the source. The original camera tape and its sidecar have separate SHA-256 digests. Every fit checks them against the label index and verifies the sidecar pre-state and future targets against the camera tape. [Replay and direct capture](waypoint_labels.py) · [Fit input checks](waypoint_data.py)

| Replay comparison | Maximum discrepancy |
| --- | --- |
| Wrist XYZ or translational motion, Euclidean norm | 0.1 mm |
| Wrist orientation or angular motion | 0.001 rad |
| Jaw opening or jaw motion | 0.1 mm |
| Paddle trace plate XY and grip-site XYZ, pre-action | 0.2 mm at every available trace sample and the final pre-action tick |

The collector rounded trace coordinates to four decimals and sampled every 20 controls. Between trace samples, the replayed object position has indirect evidence from identical native actions, aligned public trajectories and the native endpoint. Direct camera capture measures the object at every pre-action state in the same simulation that produced the frame. A divergence records its first failed comparison and the verified prefix maxima. With `--camera-fallback commands`, the original camera collector tries the stored controls once. With `commands-and-route`, it can also try the original source route once. A positive fallback publishes **its own** frames, issued actions, terminal state, reset and auxiliary labels. A failed attempt contributes zero training frames. The direct fixed-route collector uses the original primary rim/28°/180°/−35° recipe. [Camera collector](paddle.py) · [Fixed cohort](waypoint_capture.py)

For a tape with `T` controls, the state array runs from the opening `s[0]` through the actual terminal `s[T]`. At every pre-action index `t`, the sidecar stores the current plate, the achieved wrist waypoint `xyz[min(t+6,T)]`, that waypoint relative to the current plate, and its displacement from the current wrist. All use simulator world axes in metres. Six ticks give a nominal 300 ms lookahead at 20 Hz. At the tail, targets stop at the real terminal state. These labels supervise movement along a successful trajectory. They cannot recover the intended contact force when the wrist is stalled.

## Policy, fitting and checkpoint accounting

Two original camera frames become two contextualized 16 × 16 Gemma patch grids. Four learned queries per view give soft spatial coordinates and pooled visual features. The visual features and instruction predict the current plate XYZ. The motor head conditions on those features, the predicted plate relative to the current wrist, and the public eight-value hand state. It predicts five consecutive six-tick wrist displacement vectors in world axes plus rotation and jaw commands. Each translation command is its corresponding predicted displacement divided by 0.035 m, multiplied by a learned positive per-axis gain shared across views, arms and steps, then clipped to native bounds. The policy replans after five controls. [Policy](waypoint_motor.py)

The loss combines plate localization, continuous displacement and native control imitation, each with unit weight. Huber residual scales are 0.10 m for the plate and 0.035 m for displacement. The native auxiliary loss calibrates the proportional gain and supervises rotation and jaw. Training averages valid slots within an anchor, then weights anchors equally. The optimizer updates the head and rank-eight query/value LoRA in Gemma layers 16 and 17. The first update requires a finite nonzero LoRA gradient. The job records the component losses and gradient norm. The supervised path uses only layouts 0–27. [Trainer](waypoint_train.py)

Source development uses exactly external RGB, wrist RGB, public XYZ/quaternion/jaw and the instruction. The inference API rejects extra fields. Body/pose identifiers, simulator geometry, route parameters, demonstration traces, action retrieval, handcrafted phases or task goals, future observations and progress counters are excluded. Source geometry is used only to produce labels. The deployment head uses the plate **prediction** during both fitting and control.

The sampler balances arms and then samples layouts, unique tapes and controls uniformly. Within each arm it orders by numeric layout and trajectory digest before selecting anchors. Auxiliary targets join by indexed path, digest and exact pre-action state. Moving or reordering equivalent bank snapshots preserves the sampled trajectories for a fixed seed. A resume requires the original indices, code, recipe and runtime. Preserve the raw and auxiliary files on project storage. The existing staging routine requires up to 0.375 MiB per frame pair.

Offline selection uses the campaign's equal-arm native action MAE on fixed layout-balanced anchors from 28–31. Each candidate also reports 3D plate error against a training-mean baseline and wrist-displacement error against zero motion, separately per arm. These metrics use metres. If localization fails to improve on the mean or if motion fails to improve on zero, the corresponding learned target has weak offline evidence. Compare the native panel even when both errors improve. Cross-fit differences can also reflect data coverage or the head design.

Each retained candidate stores the exact update, LoRA, head, normalization, base-weight identity and fit ID. The native evaluator validates the shortlisted candidate bytes and evaluates each candidate on layouts 32–35 per arm. It ranks by worst-arm native success count, total successes, offline MAE and earlier update. A passing winner needs at least 3/4 **on each arm**. Only that winner is copied to `selected/checkpoint.pt`. A below-gate run exits 2, records `best_dev_step`, and leaves `selected_step` and the promoted hash null. The last training update and offline winner have their own identities. The fit saves optimizer, sampler and RNG restart state at update one, every 25 steps, each retained step and the final update. To resume, repeat the train command with the same fit and arguments, a fresh launcher attempt name and `--resume`. Repeating the development command checks its existing receipts and fills missing starts.

For downstream source control inside the qualified policy interpreter, use `from waypoint_motor import WaypointMotor`, then `model = WaypointMotor.load(fit / "selected", weights)` and `actions = model.act(public_obs)`. This loader checks the passing source report, selected update, digest and base identity. The deliverable covers source development. Confirmation and transfer require registration of this architecture in their loaders and their existing independent gates.

Local verification covers source parsing, lint, formatting and shell-command syntax. Replay acceptance, localization error, GPU throughput and fresh-start success require the Linux jobs above.
