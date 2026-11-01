# First native AD/DPT fits — completed

Baige `job-ielg6papl3hn` finished successfully at 22:24 Shanghai on 2026-10-30, exit 0. All four independently initialized models (AD full/static and DPT full/static) completed 100 AdamW updates, with finite policies and complete optimizer/sampler/Python/NumPy/Torch/CUDA RNG recoveries. Each reused 1,539 source decisions for 39,621 supervised positions. No new simulator actions or VLA queries were used.

The shared source bank contains 30 complete SAC K4 trials, 120 episodes and 47,602 actual source actions. Validation contains six independent PPO source-validation K4 trials, 24 episodes and 10,552 actions. DPT labels use the same frozen SAC5 teacher on both partitions. No target data enters fitting or normalization.

Actual scheduler running time was 392 GPU-seconds on one GPU; launcher time was 360.12 seconds. The 1,200-second reservation is closed and its remaining 808 seconds are sealed. Code pin `1aeea4f9fcc307bb147def92374bb81a3e2cb50e`; independently verified portable final model identities are in [receipts/final-models.json](receipts/final-models.json), with full recovery checks in [receipts/summary.json](receipts/summary.json).

These are offline fitting results. Frozen robot success, deterministic teacher competence and a history advantage remain unevaluated. AD and DPT losses have different targets and are not directly comparable.
