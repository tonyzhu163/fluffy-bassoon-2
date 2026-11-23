# Can joint encoders recover the proprioceptive input to frozen SmolVLA?

**A pooled joint-to-pose adapter reproduced the frozen policy's known-TCP actions more closely on all 16 held-out frames.** Across eight tasks and two simulated arms, mean first-five-action RMSE fell from 0.1670 with raw joint slots to 0.0143 with the adapter. The paired reduction was 0.1527 (95% task-bootstrap interval 0.1233–0.1829). Supplying the training mean for state gave 0.1001 RMSE, a paired reduction of 0.0858 (0.0540–0.1190) with the adapter. Training covered both bodies on separate tasks. Native success and transfer to a body absent from calibration remain unmeasured.

| State supplied to SmolVLA | UR5e | IIWA | Task mean (95% bootstrap interval) |
| --- | ---: | ---: | ---: |
| Known TCP reference | 0 | 0 | 0 |
| Logger's TCP | 0.1184 | 0 | 0.0592 (0.0483–0.0699) |
| Raw joint slots | 0.0995 | 0.2346 | 0.1670 (0.1393–0.1989) |
| Joint-to-pose | 0.0127 | 0.0159 | 0.0143 (0.0063–0.0233) |
| Drop state | 0.1042 | 0.0960 | 0.1001 (0.0635–0.1376) |
| +90° replay, raw XY | 0.1528 | 0.1603 | 0.1566 (0.1243–0.1798) |
| +90° replay, fitted XY | 0.0042 | 0.0039 | 0.0040 (0.0027–0.0054) |
| −45° replay, raw XY | 0.0806 | 0.0756 | 0.0781 (0.0565–0.1024) |
| −45° replay, fitted XY | 0.0041 | 0.0038 | 0.0039 (0.0027–0.0053) |

For each task, average the two arms' errors, then resample eight whole tasks with replacement 10,000 times (seed 915). The intervals are percentile bounds over the sampled object instructions. Each task contributes one frame per arm and one paired noise draw. The intervals do not capture variation across trajectories, noise draws or new task families. [Task scores, input hashes and paired differences](scores.json)

The joint adapter's median TCP position error was 2.90 cm and its median rotation-vector error was 0.0485 rad. Its advantage persisted over the entire unclipped 50×7 chunk (mean RMSE 0.0346, versus 0.3966 for raw slots and 0.3975 for dropped state). The two coordinate replays isolate XY frame error on fixed recordings. Archived support estimated +91.86° and −46.80°, giving paired reductions of 0.1525 (0.1209–0.1751) and 0.0741 (0.0538–0.0971), respectively.

On one held-out UR5e frame, shifting any one of the eight state channels by one source standard deviation changed the predicted action chunk. The largest response was to TCP Y (0.1798 RMSE) and the smallest was to rotation-vector Y (0.0090). The unperturbed action and noise digests matched the main run in a separate process. The reference action varied over time in all seven components on every evaluated frame. Compute used 0 GPU-hours. [Channel probe](dimensions.json)

## Comparison

The primary experiment uses recorded simulated UR5e and IIWA trajectories. Each frame contains two synchronized cameras, measured joints, TCP pose and two finger positions. The oracle uses the measured TCP, a consistent rotation-vector branch and measured fingers from **that same frame on that same arm**. The question is whether paired calibration samples let a joint-input adapter approach the oracle's output more closely than direct joint-slot substitution or withholding proprioception. This comparison measures frozen forward-pass consistency. A native reward or successful grasp would require a rollout.

| Input | Information available at inference |
| --- | --- |
| Known TCP | Measured position (3), axis-angle (3) and finger positions (2). Use the archived downward-gripper hemisphere rule. |
| Reported TCP | The same eight logged values, including the logger's original rotation-vector branch. |
| Raw joint slots | The first six joint angles in the six pose slots and both measured finger positions. SmolVLA has no seventh arm-joint slot. This is a direct-wiring control. |
| Joint-to-pose | All available angles (6 UR5e or 7 IIWA) and measured fingers. A single calibration memory maps angles to canonical pose. |
| Drop | The checkpoint state mean, which becomes exactly zero in every normalized state channel. The policy still receives its state token. |
| Raw or fitted local XY | Apply the archived reset-anchored local position convention to the known TCP. Restore XY using yaw estimated from three historical command–motion pairs. Test both interfaces on both arms, preserving the other six channels. |

The frozen motor consumes the 8D state and zero-pads it to 32 before a learned linear projection. It sees two RGB views and a language instruction. It receives no joint/arm ID, command-interface label or target action. Its decoded 7D relative OSC proposal has three translation, three rotation and one gripper component. The official config uses 50 steps per chunk and 10 flow-integration steps. The primary metric takes the first five decoded actions, clips to [−1, 1] per component as in the archive and computes 35-component RMSE against the known-state proposal. The scores also retain raw 50×7 chunk RMSE. No command is issued. [Motor conversion](../../libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py) · [State and action transforms](../../../src/fluffy_bassoon/xemb/embodiment.py) · [Archived action and quaternion contracts](../../symmeta-fk-crossarm-20261123-r1/QUERY_CONFIG.json)

### Calibration and sampling

- Pool 288 timestamp-aligned encoder/pose pairs across both robots from LIBERO-Object tasks 0 and 1 (orange juice and cream cheese), three episodes per task per robot and 24 evenly spaced frames per episode. Use the sine and cosine of seven angles, padding an absent trailing encoder with zero radians. For each query, fit a local affine map on its 32 nearest calibration samples with ridge penalty 0.001 on slopes only. Copy both finger readings. This fit needs measurements of Cartesian pose during target calibration. It uses no arm label and never inspects a test-task pose to fit coefficients.
- Test tasks 2–9, one midpoint from the first episode on each arm per task, specified by task and episode indices before seeing model outputs. These are eight held-out object instructions in the same pick-and-place family. Both bodies appeared in calibration on tasks 0–1, and their public encoder vectors can distinguish them. Each task gets one deterministic noise draw shared across all conditions and both bodies. Both recorded views and the prompt stay fixed within each comparison.
- The UR5e logger reports a negative X rotation-vector component at the sampled frames. Every source-training example had a positive X component according to the checkpoint's saved minimum (+0.353 rad). Apply the archived downward-gripper quaternion sign convention as an equivalent rotation vector, changing the sign branch without changing orientation. The IIWA samples already use the positive branch. The logger's Cartesian positions are retained. Midpoint Z spans 0.075–0.313 m on the two bodies, below the checkpoint's source first percentile of 0.636 m. The reference defines a state-channel contract. World-frame registration to the source robot and action quality remain unverified.
- The frame replay uses +90° (UR5e historical fit) and −45° (IIWA historical fit) from the archive. Each estimate uses three original clipped issued-XY exposures and their FK world-XY displacements from LIBERO-10 task 5. The interfaces' specified angles only construct the replay and validate the inverse. The estimator receives numerical control and displacement pairs. Neither gain nor gripper sign acts on the position observation. These shifts are counterfactual on the public recordings. [Archived FK inputs](../../symmeta-fk-crossarm-20261123-r1/RESULT.md) · [Procrustes implementation](../../symmeta-camera-command-20261123-r1/world_control.py)

## Reproduction

Run code revision `35f31650e66558198ef23f6f59414420cf2ecb7c` starts from campaign baseline `757a44cd4374833cad790852df8b87ddf412e638`. The scripts and asset hashes are also carried with the scores. Use a Linux x86-64 CPU and CPython 3.12.16. Run the following in the state run directory.

```bash
assets="$HOME/state-exp"
uv venv --python 3.12.16 "$assets/venv"
uv pip install --python "$assets/venv/bin/python" torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$assets/venv/bin/python" -r requirements.txt
bash fetch.sh "$assets"
manifest="$PWD/assets.sha256"
(cd "$assets" && sha256sum -c "$manifest")
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1
"$assets/venv/bin/python" run.py "$assets" --out scores.replay.json
"$assets/venv/bin/python" dimensions.py "$assets" --out dimensions.replay.json
```

The release contains the full fine-tuned VLM and action expert. The loader constructs the architecture from the pinned SmolVLM config, strictly loads all policy weights and uses float32 CPU inference in eval mode. The model has 450,046,176 parameters, 0 trainable. Its safetensors SHA-256 is `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`. The saved pre/postprocessor pipelines perform the camera renaming, task newline and tokenization, batch conversion, normalization and action unnormalization. Only device and tokenizer location are overridden. The LeRobot PyAV reader returns recorded RGB channel-first images in [0, 1]. The agent view is already upright in these datasets. The policy applies its configured 512×512 resizing and vision scaling. [Checkpoint](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d) · [VLM config and tokenizer](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/4cde9d208e19519c36917aa170fffc7e381aa0ad) · [UR5e recordings](https://huggingface.co/datasets/OliverHausdoerfer/libero_object_ur5e_defaultCams/tree/473f94e10c19660ccf89bf434e283c782c9c8c32) · [IIWA recordings](https://huggingface.co/datasets/OliverHausdoerfer/libero_object_iiwa_defaultCams/tree/8aaff5f95753f00d15b51d8046efbffcc6446189)
