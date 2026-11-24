# Can image-only roll correction recover a frozen SmolVLA action proposal?

[Paired results and limitations](RESULT.md)

## Comparison

An input adapter estimates a discrete mounting roll independently in each camera and rotates its image before SmolVLA preprocessing. It requires a known nominal orientation at calibration. A failure to reduce paired proposal error on held-out tasks, or a comparable disturbance to correctly mounted cameras, would count against using this adapter.

The [prior camera study](../../libero-xemb-camera-20261119-r1/RESULT.md) found that reset appearance exposes physical agent-camera roll on seven of eight development tasks. We borrow its task IDs, 16×16 RGB template design and quarter-turn intervention. Here we fit new templates and query an actual two-view motor on freshly rendered **standard LIBERO Panda** states. The earlier camera audit committed hashes for the synchronized wrist and proprioception. This experiment tests two camera mounts on Panda, with native success and transfer to a different arm still open.

* Eight fitting tasks supply one labeled-upright frame per camera from reset 0. Each sensor's template is the mean of the eight downsampled images. On each query, independently try 0°, −90° and +90° array rotations for each camera and minimize squared RGB error to its own template. Both sensors are corrected, even when only one mount was disturbed. No query pairs, task identity, pose readback, rewards, actions or camera labels enter this estimator.
* Eight disjoint development tasks, two per suite, each supply reset 13. Every reset settles for ten dummy controls. The agent and wrist cameras each undergo −90° and +90° rigid rotations around their own local forward axis, one sensor at a time. MuJoCo rerenders each view. We verify relative rotation, fixed camera center, unchanged integration state, public proprioception and the other camera's pixels, and an exact return to nominal. These are physical changes in extrinsics. Only inverse correction uses an array rotation. The perfect-transform control uses the known roll on the shifted render and can retain rasterization differences.
* The reference is the nominal frozen proposal at the same observation, state, instruction and noise. Report raw shifted, estimated correction and perfect correction relative to it. Also apply the estimator to a nominal input to reveal false corrections. Inference uses one independent standard-normal draw per task, explicitly cloned for each condition. Rerun the first nominal input and change its noise in a separate check.
* Primary readout is RMSE of the entire unnormalized 50×7 action chunk in LIBERO command units. Each task's two signs receive equal weight for each sensor. Bootstrap 20,000 whole task pairs with replacement for a descriptive 95% percentile interval. Secondary readouts separate translation, rotation, gripper, first control and first five controls. The paired improvement is raw minus estimated RMSE, positive for an improvement. A single reset and noise draw per task limit generalization across states and sampling noise.

## Reproduction

The repository baseline is `757a44cd4374833cad790852df8b87ddf412e638`. Use the factory commit containing this directory. On Linux x86-64 with Mesa EGL/llvmpipe, uv, curl and unzip, run the following command from this directory. Scratch stores simulator assets, images, environments and model weights outside Git.

```sh
bash reproduce.sh "$HOME/view-scratch" "$HOME/view-result"
```

The run uses Ubuntu 22.04.5, Mesa 23.2.1, CPython 3.10.12 and 3.12.16, and uv 0.13.9. The [setup script](reproduce.sh) pins the upstream LIBERO source and full assets, and the two requirements files pin the installed simulator and motor dependencies. The [download manifest](downloads.sha256) covers the policy, preprocessing statistics and tokenizer/config files. The reset schema and per-array hashes are in the render manifest. Different graphics drivers can change rasterization.

SmolVLA uses the strict full `lerobot/smolvla_libero` checkpoint at `a24f1f380461b665158d52acd140152c79e8b36d` with `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` tokenizer/config at `4cde9d208e19519c36917aa170fffc7e381aa0ad`. CPU inference uses BF16 and FP32 parameters, with all parameters frozen and all modules in evaluation mode. The model initializer uses the base architecture only, and strict loading supplies every trained weight from the full checkpoint. LeRobot 0.4.4 handles uint8 image scaling and batching, LIBERO's 180° image flip and position/axis-angle/gripper state, camera-key renaming, newline and tokenization, normalization, and action unnormalization. Its standard 512-pixel image resize and 10 denoising steps are preserved. The policy receives the official task string. Repeated identical image pairs within a single task/noise draw reuse the byte-identical input's proposal.
