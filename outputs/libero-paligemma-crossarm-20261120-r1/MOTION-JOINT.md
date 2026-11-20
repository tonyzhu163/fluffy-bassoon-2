# Stage 2 source motion matching

Can a practice-conditioned reader and action adapter reduce Sawyer’s predicted motion mismatch on actual Panda-trained motor proposals while preserving Panda’s predicted response? The registered two-source expert gate failed when 0 of 16 Sawyer task demonstrations succeeded. This diagnostic uses Sawyer’s measured response to prescribed commands. Successful Panda demonstrations supplied the motor’s task supervision. [Registered protocol](EVAL.md) · [Panda control](PANDA-CONTROL.md)

## Inputs and source resets

Run from this directory on the qualified Linux GPU and native simulator runtime. Supply the full reader01 Stage 1 directory, including its practice tapes, and a fresh output directory.

```bash
python motion_joint.py \
  --motor-output "$RUN_ROOT/panda01/motor" \
  --weights "$VLM_WEIGHTS" \
  --pretrain "$RUN_ROOT/reader01/pretrain" \
  --out "$MOTION_JOINT_OUT" \
  --revision "$CODE_REVISION"
```

| Input | Use |
| --- | --- |
| Panda-only motor | Audit the completed 300 updates and load the validation-selected update 150 with the original verified PaliGemma weights. |
| Stage 1 reader | Load the full update-100 state and require SHA-256 `f2add8aa19135a6841d0970bbeb496951714a6e03bd255579ed4664a95ac07e3` in both the file and completed stage report. |
| Source poses 20, 21 | Verify both arms’ tapes against that report. Encode each 16-transition history separately and retain the seven Stage 1 motion scales. |
| Source poses 22, 23 | Verify the disjoint tapes and fit one local action-to-motion map per arm from its 32 measured transitions. |
| Source poses 24, 25 | At pre-action times 0, 5, 10 and 15, call the frozen motor with the native external and wrist images, proprioception and instruction. Execute the pose-seeded 16 prescribed commands. Cache all five rows of each proposal, including the last chunk. |

## Objective and controls

For each source arm `b`, fit the seven-dimensional affine map `M_b(a) = a @ G_b + d_b` in float64. Divide each observed motion coordinate by its Stage 1 source RMS scale. Minimize the sum of squared residuals plus 0.01 times the sum of squared gain coefficients, with a free intercept. The fit covers all seven command and response coordinates, including the gripper. The report records the fitted bases, intercepts, observed command ranges and per-axis fit residuals.

Each update crosses the two supports per arm with all 40 cached proposal rows for that arm. The resulting 160 equally weighted examples include 80 per arm. For a proposal `u`, the reference is `M_Panda(u)`, evaluated on that same proposal even when Sawyer supplied the observation. Compute the unconstrained action with `ReaderAdapter.adapt(encode(history), u)`. Clamp to the controller interval `[-1, 1]` before applying the arm’s map. The loss adds three terms, each averaged across examples and coordinates.

| Term | Weight |
| --- | ---: |
| Squared difference between the adapted arm response and the Panda reference | 1 |
| Squared distance from the unconstrained action to its clamped value | 1 |
| Squared distance from the clamped action to the original proposal | 0.01 |

Panda receives the same objective, with its own unmodified motor response as reference. Fit only the actual reader and adapter parameters for 150 full-batch Adam updates at 0.003, with gradient norm capped at 5. Context dropout is fixed at zero so every fitting example uses an observed support. Cleared and pose-matched wrong-arm histories are scoring controls. The Stage 1 response head and motion scales stay fixed.

## Reading the evidence

`report.json` gives the joint loss and its components before each optimizer step, pre/post per-arm motion MSE on identical cached proposals for all three contexts, and the L2 distance each trained module moved from its Stage 1 parameters. Failure to reduce Sawyer’s correct-context MSE counts against successful source fitting. A rise from Panda’s identity baseline measures the cost to Panda. Similar scores for correct and control histories weaken the case that the correction uses arm-specific practice.

The score measures agreement under local affine maps. Its proposals revisit fitting inputs, and some commands can exceed the range explored by the response tapes. Physical task success requires a separate rollout. This diagnostic leaves the failed registered gate in force.

The proposal archives retain the pre-action observations, chunks and seeds, reset fingerprints, prescribed commands, measured transitions and native rewards. The report records the implementation and input digests, configuration, motor checkpoint hashes before and after training, and counts of trainable motor parameters and populated motor gradients. The final `reader-adapter.pt` carries the version-1 state dictionary. Completion requires exact tensor readback through `adapt_eval.load_reader`. The parent owns launch and evaluation.
