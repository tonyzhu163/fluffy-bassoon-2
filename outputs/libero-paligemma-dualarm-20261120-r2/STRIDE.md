# Does Goal 5 benefit from replanning sooner?

The reported stride-5 development counts are Panda **2/4** and Sawyer **1/4** at SFT update 1600, and **0/4** and **2/4** at update 800. The cheap comparison fixes update 1600 and varies only how many commands the motor executes before it sees the cameras again. An increase in paired native successes on the weaker arm without losing the stronger arm would support pursuing shorter execution. A change in approach distance alone cannot clear the development gate. Four layouts per arm support a development decision. Independent confirmation still requires 6/8 per arm.

## One development pass

Use the qualified R2 Linux policy environment. Point `FIT` at the first 1600-update fit, including its original `dev`, `manifest.json`, `candidates.json` and candidate files. The ordinary R2 launcher supplies the pinned code revision, policy weights, simulator interpreter and rendering environment. Both commands below use working-directory-relative output names within separate fresh attempt directories. [Launcher](run.sh)

```bash
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
FIT="$campaign/sft01/fit"

bash "$run/run.sh" "$PIN" stride01 vlm stride.py \
  --fit "$FIT" --step 1600 --strides 1 2 --out panel

bash "$run/run.sh" "$PIN" stridereplay01 vlm stride_replay.py \
  --report "$campaign/stride01/panel/report.json" --out geometry
```

The first command executes **16 fresh raw episodes**, once per stride on Panda and Sawyer layouts 32–35, and reads eight held stride-5 tapes. It accepts a completed, below-threshold development report. It checks candidate bytes and the original report, contract, reset and native action/reward tapes before loading the model. The existing `motor_eval.FrozenMotor` dispatches through the R2 servo architecture factory, checks the base snapshot, restores the selected update and freezes it. Historical checkpoints use the original chunk head. Only include `5` in `--strides` when budgeting eight additional fresh episodes to reproduce the original action and outcome streams. A completed comparison exits zero for any observed success count. [Source CLI](stride.py) · [Frozen loader](motor_eval.py)

Each episode has the same 300-control ceiling. A full episode at stride 1, 2 or 5 costs **300, 150 or 60 motor calls**, respectively. Early termination reduces both counts. The report gives actual controls and calls per episode and arm, wall time per fresh episode including reset and tape handling, and paired win/loss layout IDs for every comparison. Held call counts are reconstructed as `ceil(controls/5)`. A gain can reflect both earlier feedback and additional inference. [Comparison and output](stride.py)

## Pairing and evidence

Each motor call receives only `rgb`, `wrist`, 8-D public state and the task prompt. The native reset seed is `20261120 + pose`. R1's RNG-preserving proposal call uses `900000 + 1000 * pose + elapsed_controls`. Replans shared by two schedules therefore have the same seed. The rollout checks the reset fingerprints, public opening and all 35 opening action values against the held episode before control zero. Subsequent observations follow the controls actually issued. At shared fresh plan times the comparator checks equal observations until the first different command and equal full proposals whenever the inputs match. [Rollout and readback](stride_rollout.py)

Each new NPZ contains the full float32 `chunks[calls,5,7]`, the executed raw prefixes `proposals[controls,7]`, the actual `actions`, motion, reward, done and success streams. The `plans` JSON records elapsed controls, noise seed, hashes of each public array, the exact prompt, the combined input hash and the full chunk hash. Array hashes cover dtype, shape and contiguous bytes. The receipt binds the model, fit, schedule, counts and stream hashes. The report binds file hashes for the original inputs, held and fresh tapes, training and running implementations, and the verified base identity. The launcher log receives the final report SHA-256. Legacy tapes supply the opening chunk and issued control stream; the fresh tapes also preserve discarded suffixes at every replan.

## Separate geometry replay

The second command opens fresh simulator sessions on the eight source development starts and replays each recorded raw action tape. It instantiates no motor. It checks exact opening, recorded plan observations where available, and motion, reward, success and done at each control. A mismatch marks the diagnostic incomplete. The policy process never calls the privileged scene API. [Replay CLI](stride_replay.py)

Each trace stores the reset and every post-control plate, TCP and goal-site position, finger contact and distances. “Closest plate” is the minimum 3-D TCP-to-plate-center distance. “Closest goal” is the minimum horizontal plate-center-to-goal-site-center distance. Each plateau window contains 12 controls and all 13 plate XY samples must be within 2.5 mm of each other. Windows may overlap. The report gives their count and eligible denominator, first window end, final and closest distances, per-layout differences against held stride 5 and per-arm medians. These measurements describe approach or stalled plate motion. Native sparse success is the selection metric.

## Integration if a stride is selected

The standalone development CLI writes only exploratory evidence. The following are the integration points for a production selection with the same candidate and rollout implementation.

1. **Motor checkpoint and training contract.** Keep `recipe["horizon"] == 5`, the five-slot supervised labels, architecture, optimizer and checkpoint tensor shapes. For new fits add `contract["execution"] = {"proposal_horizon": 5, "candidate_strides": [1, 2, 5]}` when building the fit manifest in `motor_train.train`. Each saved candidate already embeds that contract. Put the selected integer stride in the post-dev selection record alongside the exact checkpoint SHA, fit ID, development report SHA and `stride_rollout.py` SHA. For the existing SFT1600, bind its original manifest and unmodified checkpoint by SHA in a versioned selection sidecar. A historical selection with no stride evidence has stride 5. A new selection using that historical checkpoint needs the explicit sidecar and paired development tapes. [Trainer](motor_train.py)

2. **Source development and confirmation.** In `motor_eval.run/score_panel`, store a versioned execution contract containing `proposal_horizon=5`, `execute=stride`, rollout SHA and plan-seed rule. For multiple settings key score rows and destinations by both checkpoint and stride. Instantiate `PairedRollout(model, stride, c["sha256"], fit_id, prompt)` and call `runner.episode(dest, body, pose, expected=opening)` on missing tapes. The runner owns its fresh native session. For readback call `check_tape` with the sealed stride, body, pose, condition, checkpoint, fit ID, prompt and recorded tape SHA. Add the stage-relative tape path, digest and step to its returned receipt for the existing report comparison. Dispatch the historical stride-5 schema to `check_episode`. Its opening-prefix check assumes all five opening commands were executed. For the old fit, verify its fit-time sources against its immutable manifest, and store current evaluator sources separately in the versioned panel. Use the numerical source checks in `stride.run` and compare current evaluator hashes to the sealed panel during readback. [Stage producer](motor_eval.py) · [Stride evidence schema](stride_rollout.py)

3. **Seal and gate.** Recount development from the eight tapes per setting. Require 3/4 per arm. Apply the existing order by minimum arm successes, total, offline MAE and update. For an exact tie on the same checkpoint prefer the largest stride to conserve inference. Bind that rule and the candidate set before selection. Include the winning stride and rollout SHA in `selection.json` and the confirmation contract before opening 40–47. Confirm just that pair. In `transfer_gate.confirm_receipt`, replace the fixed `panel["execute"] == ev.EXECUTE` in the new schema with agreement between selected stride, dev winner, seal, panel, per-episode receipts and reports. Verify proposal horizon five, the bound evaluator implementation, historical training provenance and the full issued-prefix checks through `check_tape`. Retain the layout-exclusion checks, all 16 native tape recounts, 6/8 per arm and the existing lock. Return the verified stride in the gate dictionary. [Transfer gate](transfer_gate.py)

4. **Reader fit and target queries.** Pass the gate's stride into `transfer_fit.cache_experts`. Change its anchor loop to `range(0, len(ep.actions), stride)`, retain the full `(5, 7)` shape check and append `chunk[:min(stride, len(ep.actions) - t)]` at each anchor. Bind the stride, rollout SHA and five-slot horizon in the Stage 2 checkpoint, `fitted_reader` comparison, `execute_prefix` report field and `bind_target` record. Use `gate["execute"]` for every target condition and keep the existing calibration/support tape checks. Stage 1's measured command–motion fit is independent of motor inference. In `transfer.evaluate` use the following calls inside the existing source-confirmed scope. [Expert proposal bank](transfer_fit.py) · [Transfer evaluation](transfer.py)

```python
runner = PairedRollout(motor, gate["execute"], gate["motor_sha256"], gate["fit_id"], gate["prompt"])
transforms = runner.transfer_transforms(net, correct, wrong, responses["Panda"], responses[TARGET])
for condition in CONDITIONS:
    expected = None if condition == "raw" else results["raw"]["opening"]
    results[condition] = runner.episode(args.out, TARGET, pose, condition, transforms[condition], expected)
```

Keep the existing post-call target prompt, release-row and native reset checks. Add `stride_rollout.py` to the evaluator implementation hash map. Correct and wrong use their respective measured histories, empty uses R1's zero reader context, and calibration uses R1's Panda-to-target response transform. Only the executed prefix is transformed. Each tape retains the full raw proposal.

## Verification status

The three Python sources were statically parsed on macOS. New native success counts, model-call timings and replay distances require the Linux commands above. The stride-5 counts in the first paragraph were supplied with the assignment.
