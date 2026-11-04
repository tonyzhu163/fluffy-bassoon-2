# Three matched OpenDrawer PPO comparisons at 3,072,000 training actions

Approved November 4. Test the effect of reversing each Stage 1 recipe change at its 40-update endpoint. The completed Stage 1 checkpoint is the common control. The human approved all three contrasts and a separate 92 GPU-hour ceiling in this chat.

| Arm | Epochs per rollout | Critic input | Eligible stochastic denoising indices | Adam steps | Training |
| --- | ---: | --- | --- | ---: | --- |
| Stage 1 control | 1 | Frozen VLM prefix | 0–8 | 240 | Reuse step 40 |
| E, rollout reuse | 4 | Frozen VLM prefix | 0–8 | 960 | Fresh, 40 updates |
| C, critic placement | 1 | Action-expert suffix at the sampled step | 0–8 | 240 | Fresh, 40 updates |
| D, last step eligible | 1 | Frozen VLM prefix | 0–9 | 240 | Fresh, 40 updates |

The three primary estimates are E, C and D minus Stage 1. They estimate each change with the other two settings fixed at Stage 1. They leave interactions and variation across independent training seeds open. The historical step-40 actor and the initial SFT actor are additional evaluation references. The historical actor was trained with a different value definition, so it cannot substitute for C.

## Training match

Use the same Human300 SFT actor, normalization, frozen weights, native simulator first-success reward, 128 fresh OpenDrawer scenes per rollout, training seeds 2,100,000–2,105,119 in original order, 600 controls per episode, H50, 10 denoising steps, noise 0.3, zero critic warmup, global chunk minibatch 256, two actor ranks at microbatch 16 and eight rollout/environment ranks. Preserve optimizer settings, masks, five DEV64 panels, checkpoint cadence and actor-batch retention. Each arm executes 40 × 128 × 600 = 3,072,000 fresh training actions or 61,440 chunk decisions. E reuses each on-policy batch four times. Each arm collects its own trajectories as its actor changes.

Before submission, check current main and diff resolved configs against the [actual Stage 1 settings](../robocasa-ppo-grpo-meta-stage1-20261104-r1/RCA-METRICS-R1.json). Pin one reviewed training runtime for all new arms and keep RLinf at the qualified selected-value implementation `8f9bfa76`. The [four-epoch profile](../robocasa-ppo-recipe-fixes-20261104-r1/RESOLVED-CONTRACTS.json) has passed Linux config composition. C needs matching actor and rollout `value_after_vlm=false`. D needs `ignore_last=false` on both workers and all ten inference denoising steps still execute in every arm. Read the effective placement and batch sizes from each resolved launch config.

Start from a verified private copy of the Stage 1 step-0 state. Verify actor weights, optimizer step zero, sampler cursors, per-rank RNG, environment seeds and the reference before panel. E and D can share the original prefix-head initialization. C requires a separately initialized suffix head of the proper shape. Compare all non-head actor parameters by hash, qualify its optimizer groups and initial state, and restore the same worker RNG after construction. Strict-load each arm into its own model architecture. If the common starting state cannot be reconstructed, stop before spending the full budget and re-plan with a fresh common control.

Use a real first collection to verify physical/public resets and the initial noise stream. E must reproduce the original first collection to the measured repeat tolerance before the extra epochs. Record the selected indices for D, including index 9, and account for sampler differences introduced by that treatment. Qualify likelihood and value replay before SGD for every arm, including the selected suffix value for C. Check finite losses, unchanged frozen weights, step counts and completed 600-control episodes for all 40 updates. Record per-epoch KL, clipping and actor/head gradient norms and parameter changes where qualified. Report observed divergence and throughput.

## Primary readout

Use only step 40 for the three primary contrasts. Evaluate all six actors on the same 512 fresh OpenDrawer cases, seeds 31,941,000–31,941,511, in fixed eight-case batches. For each bank, use the same initial physical/public state, identical lane membership and a common sampler RNG seed of 41,941,000 plus the zero-based bank index. Counterbalance policy order across banks. Verify actual sampler hashes, strict model loads, case identities and physical/public states. Repeat the SFT actor on the first bank and compare its decoded actions and outcomes to quantify evaluation repeatability. The six full panels and the repeat use 1,848,000 policy controls plus resets, separate from training.

The primary outcome is first success within 600 controls. Report absolute rates, paired wins and losses and paired 95% intervals in percentage points. Apply Holm correction to the three two-sided exact McNemar tests at family alpha .05. Record end-of-episode success, original-versus-Stage-1 and SFT contrasts as descriptive. Four epochs have evidence in their favor here if E increases paired success and passes the adjusted test. A wide interval spanning gains and losses leaves the epoch choice unresolved. No DEV-based checkpoint selection.

## Incremental allocation proposal

The prior one-epoch arm cost 17.13 GPU-hours all-in. The historical four-epoch cadence was 257.32 seconds per update on eight GPUs, about 22.9 GPU-hours for 40 updates before startup and evaluation. Use 17–19 GPU-hours per one-epoch arm and 24–27 for E as planning estimates. Recent paired evaluation cost 7.00 GPU-hours for 2,048 valid policy episodes including setup and recovery. The 3,080 proposed evaluation episodes forecast roughly 10.5 GPU-hours at that observed all-attempt rate. Exact timing for C and E needs on-device verification.

Approved hard cap of **92 incremental GPU-hours**, comprising 32 for E, 24 each for C and D, and 12 for the paired evaluation including qualification. This includes device admission, setup and release. Train serially unless storage and capacity independently permit overlap. Run the epoch arm first, verify first-update parity and measured time/storage before continuing, preserve complete boundaries, and stop at the caps. New training totals 9,216,000 actions across three arms. Keep accounting separate from the original Stage 1 PPO/GRPO approval. No automatic retry, new seed, expanded factorial or training extension.

[Stage 1 audit](../robocasa-ppo-grpo-meta-stage1-20261104-r1/PPO-AUDIT-LIVE.json) · [timing comparison](../robocasa-ppo-grpo-meta-stage1-20261104-r1/PERFORMANCE-NOTE.md) · [paired endpoint evaluation](../robocasa-endpoint512-acceleration-20261104-r1/RESULT.md).
