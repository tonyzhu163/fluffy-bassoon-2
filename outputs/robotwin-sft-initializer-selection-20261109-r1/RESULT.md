# RoboTwin SFT initializer selection

Recommend **SidneyXie/pi05_robotwin**, pinned to
`e49e2ab6c11f07511573b67261bd129e88d0a416`, as the common broad RoboTwin
π0.5 SFT initializer to qualify before PPO/GRPO.

The complete original nine-file bundle is now downloaded on Baige and every
file is verified against its upstream Git blob or LFS digest. No conversion or
optimizer update occurred. The model is9,354,050,752bytes, SHA256
`9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3`.

Path: `/mowenjie/fluffy-bassoon-2/cache/sidney-pi05-robotwin-e49e2ab6-original-20261109-r1`.
The prior `robotwin-sidney-rlinf-20261105-r1` directory contains a distinct
7,774,043,776-byte converted weight file and processor summary. It is not the
original bundle just downloaded. Preserve both; do not silently substitute it
for the direct release when establishing the SFT baseline.
[Verification receipt](SIDNEY-ORIGINAL-BUNDLE.json).

## Training and inference provenance

The [model card](https://huggingface.co/SidneyXie/pi05_robotwin) and saved
[training config](https://huggingface.co/SidneyXie/pi05_robotwin/blob/e49e2ab6c11f07511573b67261bd129e88d0a416/train_config.json)
document fine-tuning from `lerobot/pi05_base` on
`lerobot/robotwin_unified`,50,000steps, batch16, seed1000, AdamW,
resume=False. The card reports27,500 demonstrations across50 RoboTwin tasks.
This is a supervised fine-tuning release, rather than the separate task PPO
checkpoints published by RLinf.

The saved model config specifies three cameras,14D state/action, absolute joint
actions, H50,10 inference steps,224x224 model images and mean/std state/action
normalization. Use its original serialized pre/postprocessors. Do not reuse
Motus H32, delta joints or quantile normalization.

The uploader reports clean phone stand66/100 and a32-task clean average67.91%.
These are reference claims, not locally reproduced outcomes. Our earlier Sidney
screen was excluded after stale scene metadata produced mismatched instructions;
it does not establish this checkpoint's SFT success rate. Bind fresh native
episode language to the actual scene before re-evaluation.

## Other candidates

- [RLinf-Pi05-RoboTwin-SFT-adjust_bottle](https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle)
  is an official RLinf task-specific SFT artifact. It is not a common phone/six-task
  initializer and would change the current task scope.
- [Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k](https://huggingface.co/Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k)
  documents50-task, clean50-per-task supervised fine-tuning for60k steps, H50,
  JAX/Orbax and quantile delta joints. It is a plausible alternative, but needs
  a different runtime/conversion qualification and its card supplies no benchmark
  success table. It was not downloaded or selected.

## Motus issue interpretation and current state

The user clarified that the concern is similar public evidence, not a new
π0.5-specific author reply. [Issue11's author comment](https://github.com/thu-ml/Motus/issues/11#issuecomment-3709804565)
raises the possibility of a wrong weight upload for the **full Motus model**.
It does not confirm a mismatch for `pi0.5_robotwin2`. The exact π0.5 evaluator
request [issue45](https://github.com/thu-ml/Motus/issues/45) had no replies at
this review. Do not turn a concern into a confirmed weight-identity claim.

Our27/100 clean,25/100 randomized result remains valid for its exact released
Motus revision and recorded reconstruction. It should not be asserted to test
the paper's identical weights/config. The user now requests a clearly documented
RoboTwin SFT initializer. Motus parity, PPO and GRPO were never launched and
remain held. No GPU is active; Epoch2 cost remains2.6772/100GPU-hours.

Next: qualify the original Sidney inference pipeline with correct scene-bound
instructions, measure SFT first, then use that same initializer/protocol for PPO
and GRPO. This selection/download adds0 GPU-hours.
