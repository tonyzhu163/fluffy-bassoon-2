# Gated dual-arm transfer on native Goal 5

Does a 16-control practice history improve a confirmed, frozen dual-arm PaliGemma motor on eight fresh UR5e plate-push layouts? The outcome is native task success. A correct-history improvement over raw, empty and wrong-body practice on the same starts would support a contribution from the measured history. Similar gains with empty practice would support an unconditional action correction. The implementation has received static checks only. R2 source qualification, joint fitting and UR5e outcomes require execution on the allocated Linux runtime. [Run plan](PLAN.md) · [Transfer CLI](transfer.py)

## Source gate

Both commands consume the R2 producer's selected checkpoint, source confirmation and pre-confirmation seal. The CLI checks the checkpoint's actual source training and validation records, the development winner on 32–35, the implementation hashes and the single selected candidate in the seal. It opens all 16 confirmation tapes, verifies their embedded motor fingerprint, compares issued controls to raw proposals and recounts native successes from the rewards. All eight released layout fingerprints for 40–47 must be distinct and outside motor fitting and selection. **Each arm must reach 6/8 with the same checkpoint.** A complete subthreshold panel exits with status 2. Missing or inconsistent evidence exits with an error. Both paths precede any target reset. [Gate](transfer_gate.py) · [Motor producer](motor_eval.py) · [Motor handoff](MOTOR.md)

The selected motor loads through R1 `Motor.load(selected, weights)` and receives the existing four public inputs through `act(obs)`. Its base snapshot must match the motor's trained identity. Under a shared checkpoint lock, the CLI checks the loaded action head, LoRA and normalization tensors against the selected checkpoint before and after each stage. It also records the checkpoint hash, gradient count and trainable parameter count. [R1 motor](../libero-paligemma-crossarm-20261120-r1/motor.py)

## Cluster CLI

From the pinned R2 code directory in Baige Distributed Training, supply the verified deployment revision in `REV`, the motor fit directory in `FIT`, and the absolute source bank directories in `PANDA_BANK` and `SAWYER_BANK`. Use fresh attempt IDs. The launcher provides the base snapshot, revision, simulator interpreter and project NVMe output directory. The motor fit must contain its `selected`, `dev` and `confirm` directories, and the original campaign seal must be accessible. [Launcher](run.sh)

```bash
campaign=/mowenjie/fluffy-bassoon-2/outputs/libero-paligemma-dualarm-20261120-r2
prior=/mowenjie/fluffy-bassoon-2/outputs/libero-paligemma-crossarm-20261120-r1

bash run.sh "$REV" transferfit01 vlm transfer.py fit \
  --motor-output "$FIT/selected" --confirmation "$FIT/confirm" \
  --stage1 "$prior/reader01/pretrain" \
  --demos "$PANDA_BANK" "$SAWYER_BANK" --out joint

bash run.sh "$REV" transfereval01 vlm transfer.py evaluate \
  --motor-output "$FIT/selected" --confirmation "$FIT/confirm" \
  --fit "$campaign/transferfit01/joint" --out ur5e
```

The full Stage 1 directory includes the completed report, checkpoint and ten checksummed native source tapes. The small repository evidence copy contains only the report and checkpoint. Omit `--stage1` to retrain with R1's source-only pretrainer after the gate. Additional accepted source bank directories can follow `--demos`. The loader stages uncompressed camera observations on output NVMe during proposal collection and checks available space first. [Stage 1 protocol](../libero-paligemma-crossarm-20261120-r1/STAGE1.md) · [Expert bank loader](motor_bank.py)

## What the source fit learns

- **Stage 1** predicts measured motion on Panda and Sawyer. The restored checkpoint must match its own completed source report and metadata, including both bodies, support layouts 20–21, motion queries 22–23, a separate check on 28 and an untouched zero action residual. The CLI validates each native tape, its command script, its layout fingerprint and its recorded digest. It accepts a freshly retrained source reader.
- **Stage 2** uses successful, native expert commands from both arms. The source bank loader validates the task, body, layout, seed, reset provenance, bounded actions, aligned cameras, state and first positive terminal reward. It deduplicates trajectories. The frozen motor proposes five controls from the public observation at each expert prefix. The final partial chunk is aligned through the final expert control. Each proposal bank links to its source tape, bank index and fingerprints.
- **Joint optimization** uses 150 Adam updates at learning rate 0.003, gradient clipping at norm 5 and seed 41. Each update uses 16 samples per arm, sampling a training layout, a unique trajectory on that layout and a control. Each label receives a different-layout source support from the same arm. A shared 1/8 context-drop mask trains the empty-context baseline. The reader and seven-axis affine action residual jointly minimize mean squared error against the actual issued expert commands. The report records each arm's loss at every update and requires both modules to change. The Stage 1 response head is frozen.
- **Crossed checks** score the final update on source expert layouts 28–31. For each query trajectory the raw proposals and labels stay fixed. The evaluator substitutes Panda history, Sawyer history and empty context and reports bounded action MSE for each, plus raw MSE. Each layout has equal weight within a query arm. Queries on 29–31 use the measured history from 28. Queries on 28 use supports 20–21. Both arms receive identical probe commands for a given support layout. The update is fixed before scoring. [Fit and checks](transfer_fit.py)

## Paired UR5e panel

Before any target calibration or practice, the evaluator binds the source motor, reader, implementation and panel to `ur5e-transfer.json` alongside the campaign source seal. An interrupted job can replay the identical bound inputs in a fresh output directory. A different reader cannot reuse that campaign's opened panel. Training, update count, donor assignment, calibration reference and conditions are fixed before target rewards.

| Condition | Inputs and issued command |
| --- | --- |
| Raw | The confirmed motor acts directly on UR5e. |
| Correct | The residual sees 16 native UR5e command–motion pairs on a separate practice layout. |
| Empty | The same residual receives a zero context and can apply its learned unconditional bias. |
| Wrong | The same residual receives another arm's measured history on the identical released practice scene with identical commands. Panda and Sawyer alternate across the eight starts. |
| Calibration | Fixed positive/negative Cartesian pulses and zero commands on layouts 48–49 independently estimate Panda and UR5e responses. A damped inverse maps the raw Panda reference response to a UR5e command. The gripper passes through. A separate random probe on 39 measures response prediction error. |

Practice layouts 8–15 pair in order with fresh query layouts 40–47. Only the issued commands and measured motions enter the reader. Practice rewards are retained as evidence. Each condition gets its own native reset with seed `20261120 + layout`, a fixed practice context and the same planning seed at each replan (`900000 + 1000 × layout + elapsed controls`). Before a non-raw condition acts, its opening camera hashes, proprioception, instruction, native physics fingerprints, horizon and entire raw motor chunk must match raw. Each condition then runs closed loop for at most 300 controls. The report includes every success, native terminal reward, paired wins and losses, and checksummed practice and query tapes. [Paired native runner](../libero-paligemma-crossarm-20261120-r1/adapt_eval.py)

## Prior evidence and limits

R1 collected 11/16 successful Panda experts and 0/16 Sawyer experts. Its Panda-only motor scored 0/2 on fresh Panda. The earlier motion-matching diagnostic used a fitted source response model for its action objective. On two fresh UR5e starts, all five controllers scored 0/2. The R1 reader's held-out source motion MSE was 1.092 with correct history, 1.097 with wrong history and 1.555 with empty context. Those results motivate the stronger source gate and native dual-arm expert labels. [R1 result](../libero-paligemma-crossarm-20261120-r1/RESULT.md) · [Stage 1 evidence](../libero-paligemma-crossarm-20261120-r1/evidence/reader01/pretrain/report.json)

The eight layouts estimate a paired effect on one simulated task. The same layout IDs occur in source confirmation and target evaluation, and UR5e appeared in earlier research. Source offline action errors evaluate expert-visited states and share the motor's offline validation partition. Live states can diverge after the first action. A source-balanced residual can learn a generic correction or respond to donor-specific quirks. Report the paired history controls and per-arm crossed errors alongside native success. The analytic calibration has extra, independent pulse data and assumes a local linear Panda response remains meaningful during UR5e contact.
