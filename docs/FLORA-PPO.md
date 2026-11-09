# Learned video reward

The current RoboCasa and RoboTwin experiments use native simulator rewards. [Flora-Rewarder](https://huggingface.co/nvidia/Flora-Rewarder) is planned for episode feedback on physical robots. Before using it there, compare its grades on real task videos with independently judged outcomes and report the cost of obtaining feedback.

Earlier RoboCasa experiments used this video-reward protocol. Their calibration and failures inform the physical-robot study; they do not qualify a new camera setup or task.

1. **Record the episode.** Save synchronized left, wrist and right RGB views in that order from reset through the last physical step. Bind task instruction, horizon, frame clock, camera order and video hashes. Do not put simulator truth, success annotations or overlays in the prompt.
2. **Calibrate before training.** Compare frozen rewarder scores with separately recorded success on a fresh, fixed native case bank. Report class counts, native headroom and AUROC. Bind the reward model, prompt, cameras and horizon; changing them requires fresh calibration. Ranking successes well does not guarantee that high scores are reliable success labels.
3. **Score before the PPO update.** Normalize the official expected 1–5 terminal grade as `(score - 1) / 4`, preserving partial credit. Bind every score to its on-policy episode. Release a batch only after all labels and recovery receipts match. Strip simulator truth from the learned-reward actor and apply the terminal reward before GAE.

[Reward boundary](../src/fluffy_bassoon/robocasa/flora_ppo.py) · [native task protocols](../src/fluffy_bassoon/robocasa/ppo.py) · [video recording](../src/fluffy_bassoon/robocasa/video.py) · [measured results](FINDINGS.md#robocasa365).
