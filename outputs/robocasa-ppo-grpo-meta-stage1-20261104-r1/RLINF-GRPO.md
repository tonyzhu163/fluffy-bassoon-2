# Which parts of this run come from RLinf

The deployed RLinf commit is `8f9bfa76`. Its GRPO advantage and actor-loss files are byte-identical to upstream `d80cf00e`. The trainer uses RLinf's `adv_type=grpo` and `loss_type=actor`. It centers reward within each eight-attempt group, divides by the group standard deviation plus epsilon and applies the masked clipped actor objective. It trains no value head. The ratio clip is ±0.2. The Stage 1 plan separately reports whether logged conditional approximate KL stays at or below 0.03. That threshold has no demonstrated calibration to OpenDrawer success.

| Setting | This OpenDrawer run | RLinf's released π0.5 LIBERO-Spatial example |
| --- | --- | --- |
| Group size / optimizer epochs | 8 / 1 | 8 / 1 |
| Ratio clip / KL coefficient | ±0.2 / 0 | ±0.2 / 0 |
| Chunk-level likelihood and rewards | Yes | Yes |
| Reward filtering | Off | On |
| Training envs × rollout rounds | 128 × 1 | 64 × 8 |
| Controls per episode | 600 | 240 |
| Executed chunk / denoising steps | 50 / 10 | 5 / 3 |
| Global minibatch | 256 | 2,048 |

TRAIN draws 16 fresh OpenDrawer scene seeds per update and runs eight attempts from each. Ground-truth first success supplies the sparse reward. In the first ten collections, 146 of 160 groups contained both a success and a failure, 12 groups succeeded on every attempt, and two failed on every attempt. There were 830 successes across 1,280 attempts. DEV uses a single attempt per case with fixed groups of eight cases for inference. The saved reset fingerprints and checkpoint/rollout audits are project additions.

[Upstream advantage](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/rlinf/algorithms/advantages.py#L89-L122) · [upstream actor loss](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/rlinf/algorithms/losses.py#L577-L611) · [released recipe](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml) · [group outcome counts](GRPO-FIRST10-GROUPS.json).
