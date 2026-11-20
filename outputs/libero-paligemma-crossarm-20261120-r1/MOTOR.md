# PaliGemma source motor

The motor maps an external RGB frame, a wrist RGB frame, the task instruction and an eight-value hand state to five successive normalized LIBERO controls. It fits successful Panda and Sawyer demonstrations for BASE `libero_goal/5`. The promotion criterion is native success on fresh starts of **each** source arm. Held-out demonstration error selects the checkpoint for that evaluation. [Run plan](PLAN.md)

## Linux command

Run from the verified project checkout inside one allocated Baige Distributed Training container. The command uses the existing Linux interpreter recorded by the encoder study, which reported PyTorch 2.11.0+cu128 and Transformers 4.53.2. Set `DEMOS` to the source collector directory and `OUT` to a fresh run output on project NVMe. Set `MOTOR_CODE_REVISION` to the deployed checkout revision for the run receipt. The manifest also records byte hashes of all three motor source files. [Interpreter](../libero-witness-encoders-20261119-r1/run.sh) · [Version receipt](../libero-witness-encoders-20261119-r1/vlm.json)

```bash
set -euo pipefail
run=outputs/libero-paligemma-crossarm-20261120-r1
py=/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python
hub=/mowenjie/.cache/huggingface/hub
mix="$hub/models--google--paligemma-3b-mix-224/snapshots/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515"
weights="${WEIGHTS:-$mix}"
: "${DEMOS:?set DEMOS to the source demo directory}" "${OUT:?set OUT to a fresh NVMe output directory}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
"$py" -Bu 'scripts/[train]gpu_probe.py' --baige-container --baige-exclusive-training \
    --purpose training --expected-count 1 --output "$OUT/gpu.json"
"$py" -Bu "$run/motor.py" train --demos "$DEMOS" --output "$OUT" --weights "$weights" \
    --steps 300 --seed 20261120 2>&1 | tee "$OUT/train.log"
```

Prefer MIX. When the complete MIX release is unavailable, set `WEIGHTS` to the fully staged PT snapshot at revision `35e4f46485b4d07967e7e9935bc3786aad50687c` before running the command. Either release can live at a custom directory path. A complete local input contains the original configuration, sharded safetensors index, three model shards, fast tokenizer JSON and tokenizer configuration. The previous witness run extracted a 437-tensor vision artifact. The three upstream model shards together occupy 11,693,947,376 bytes for either release. [PT extraction receipt](../libero-witness-encoders-20261119-r1/pt-weights.json) · [MIX upstream metadata](https://huggingface.co/api/models/google/paligemma-3b-mix-224/revision/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515?blobs=true) · [PT upstream metadata](https://huggingface.co/api/models/google/paligemma-3b-pt-224/revision/35e4f46485b4d07967e7e9935bc3786aad50687c?blobs=true)

The loader identifies the release from the configuration bytes, verifies every model shard against its official LFS SHA-256 and verifies the index and tokenizer files against their upstream file identities. It passes the verified files to the local Transformers loader and checks its loading report before attaching fresh modules. [Offline verifier and pinned identities](motor_weights.py)

## Motor and supervised objective

The design follows the original pipeline's camera ordering, antialiased bicubic resize to 224, pixel range of −1 to 1, BOS and newline, one-indexed positions and bidirectional image/text prefix. The complete base passes both images through SigLIP, the multimodal projector and all 18 Gemma layers. Each camera contributes 256 tokens in raster order. Rank-eight LoRA updates the query and value projections of Gemma layers 16 and 17, counting from zero. The backbone uses BF16 and the optimizer maintains FP32 adapter and head weights. The frozen prefix runs online, so training and inference share the same observation path. [Original encoder](../../../fluffy-bassoon/src/lerobot_policy_ctm/shared/paligemma.py) · [Original processor](../../../fluffy-bassoon/src/lerobot_policy_ctm/shared/processor.py) · [Original model pin](../../../fluffy-bassoon/src/lerobot_policy_ctm/shared/configuration.py)

The fresh head keeps a separate 16 × 16 contextual patch grid for each camera. A shared convolution produces two 8 × 8 feature maps. Every location has its own input position in the action readout. The readout also receives the final valid instruction token and the normalized xyz, quaternion and jaw state. State statistics use training poses only. The target at control `t` is the issued action sequence beginning at `t`. A mask excludes controls beyond the episode boundary. The loss averages Huber errors over seven axes, valid controls per anchor and the eight anchors in the update. Inference clips each action axis to the native range [−1, 1]. [Implementation](motor.py)

Each update draws four anchors per arm. Within each arm, the sampler chooses an episode uniformly, then a pre-action frame uniformly. The default microbatch is four with two backward passes per optimizer update. `--microbatch 1|2|4|8` changes execution capacity for a new fit while preserving the global batch of eight and the loss weights. AdamW uses constant learning rates of 0.0001 for LoRA and 0.0003 for the head, weight decay 0.01 and gradient clipping at norm 1. The initial update records the measured LoRA gradient norm. The default 300 updates are an exploratory budget for one PRO 6000D. Elapsed time and memory require measurement on the allocated GPU.

## Data and checkpoint receipts

The loader reads `DIR/<body>/<pose>.npz` with pickle disabled. It checks both source arms, dimensions, finite controls and motion, unit quaternion, action bounds, zero intermediate rewards and a positive terminal reward. Embedded body, pose and success fields must agree when supplied. For the run collector, its `index.json` also verifies the compiled arm identity, reset pose and seed, native success, terminal flag, step count and final reward. With a bare archive, the source evidence is the directory, optional embedded metadata and recorded native rewards. The head receives only the four public observation fields. Decoded RGB arrays are temporarily memory-mapped under the output filesystem. Allow two raw camera streams of scratch space, about 0.375 MiB per 256 × 256 frame pair. [Data loader](motor_data.py) · [Collector](demos.py)

The seeded split holds each pose out across both arms and requires at least two successful distinct poses per arm. It targets 20% of distinct poses with at least one training and one validation pose per source arm. Validation uses up to 16 fixed anchors per held-out episode and reports per-axis mean absolute error for each arm. The lowest equal-arm mean error wins. Selection occurs at update 1, every 50 updates and the requested final update. Use the union in `manifest.json → contract.all_motor_poses` to exclude every pose used in motor fitting or selection from independent native starts.

| Receipt | Contents |
| --- | --- |
| `manifest.json` | Command, seed, source hashes, base repo/revision/file identities, per-episode hashes and split, held-out anchors, training recipe, runtime and trainable parameter counts. |
| `curve.json` | Per-update loss, gradient norm, synchronized update seconds, frames per second, peak CUDA allocated and reserved bytes, validation measurements, selected update and checkpoint SHA-256. |
| `checkpoint.pt` | Latest and validation-selected LoRA/head weights with training state normalization, optimizer, completed update, sampler state and Python, NumPy, PyTorch CPU and CUDA RNG states. |
| `gpu.json` and `train.log` | Scheduler GPU probe and trainer console output from the command above. |

Checkpoint replacement is atomic at update 1, every 25 updates and the requested final update. Resume the trainer command against the same data, code, runtime, seed, microbatch and output, adding `--resume`. The step count is the total desired update count. Resume restores the latest optimizer step. Inference loads the selected validation step. Extending a completed shorter fit leaves its earlier final validation eligible for selection.

```python
# With the run directory on sys.path in the policy interpreter
from motor import Motor

motor = Motor.load(output_dir, weights_path)
actions = motor.act({"rgb": rgb, "wrist": wrist, "state": state, "prompt": prompt})
# actions is float32 NumPy [5, 7], in the demonstration's native controller units.
```

## Implementation receipt

All three Python modules compiled on macOS. Ruff 0.17.5 passed lint and formatting checks, the Linux command passed `bash -n`, and the relative source links resolved. An independent static readback matched 11 of 11 recorded file identities for MIX and 11 of 11 for PT against the two pinned upstream metadata responses. Linux model loading, update timing, peak memory, resume behavior and native source-arm success await execution.
