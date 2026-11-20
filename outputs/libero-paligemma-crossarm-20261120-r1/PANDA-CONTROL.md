# Panda-only SFT feasibility control

On gpu01 the demonstrator succeeded on 11 of 16 Panda poses and 0 of 16 Sawyer poses. The control asks whether fitting those Panda demonstrations yields any native success on two unseen Panda starts. A score of 0/2 counts against feasibility under this training budget. A positive score demonstrates success on the specified starts and leaves cross-arm transfer untested.

Use the gpu01 motor demonstration directory and a fresh motor output with the main trainer arguments, adding `--source-bodies Panda`. The seed is 20261120, with 300 updates, microbatch four and eight Panda anchors per update. The default two-source fit draws four anchors per arm. Both use the same released-pose holdout, optimizer and checkpoint selection rule. The Panda fit requires accepted native collector receipts for its tapes.

After training, run the following in the qualified motor and simulator environment with a fresh evaluation output directory.

```bash
"$MOTOR_PYTHON" -Bu "$run/source_control.py" --motor-output "$MOTOR_DIR" \
    --weights "$VLM_WEIGHTS" --out "$CONTROL_OUT" --revision "$CODE_REVISION" --open-loop
```

The frozen validation-selected motor acts on BASE Goal 5 Panda poses 30 and 31 with seeds 20261150 and 20261151. Each rollout writes its native trace and updates `report.json` with the score, checkpoint SHA-256, selected update and base revision. The optional open-loop field reproduces the selected update’s seven action-axis errors from the demonstration holdout. Report the native score out of two after both episodes finish. The original two-arm gate still requires independent success on each source.
