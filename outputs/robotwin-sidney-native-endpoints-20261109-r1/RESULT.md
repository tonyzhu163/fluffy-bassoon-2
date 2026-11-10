# Sidney phone-stand baselines under corrected wrist rendering

Each cell gives successes on the same 100 preselected scenes per domain. The PPO and GRPO columns use the frozen actors from Adam update 40 of the November 6 training runs. This readout used the original LeRobot K10 sampler, absolute joint actions, H50 chunks, the native per-target controller, a 400-target limit and 1 cm wrist clipping throughout.

| Domain | Original SFT | PPO40 | GRPO40 |
| --- | ---: | ---: | ---: |
| Clean | 75 | 77 | 78 |
| Randomized | 87 | 88 | 90 |

On clean scenes PPO recovered six SFT failures and lost four successes. GRPO recovered seven and lost four. On randomized scenes PPO recovered three and lost two. GRPO recovered five and lost two. In the direct PPO–GRPO comparison, clean had seven PPO-only and eight GRPO-only successes. Randomized had zero PPO-only and two GRPO-only successes. Every two-sided exact paired comparison had p ≥ 0.45. The measured differences are small relative to the uncertainty on these panels. They describe one saved actor per method trained with the earlier deployment. [All paired outcomes](PAIRED.json), [SFT camera experiment](../robotwin-sidney-gap-20261109-r1/RESULT.md).

Each reconstructed native actor re-encoded to all 667 saved RLinf actor tensors exactly at the export dtype. Both differed from their original SFT initialization on 190 RLinf tensors. The native bundles retained the original preprocessing and separately stored input embedding and output head. The value head was omitted for frozen PPO action inference. Evaluation took zero optimizer steps. [Deployment contract](PLAN.md).

The independent readback verified 1,553 saved action chunks and exact initial images, state and prompt in all 400 PPO and GRPO episodes against the corrected SFT references. Both workers exited successfully and their Baige GPUs were released on November 9. [PPO verification](PPO40-READBACK.json), [GRPO verification](GRPO40-READBACK.json).
