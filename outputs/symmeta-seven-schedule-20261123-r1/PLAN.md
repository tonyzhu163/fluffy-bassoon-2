# Seven-shift Perceiver schedule control

Does the lower peak learning rate with warmup reduce the Stage 1 Perceiver’s yaw error on the recorded, held-out UR5e shifts? A reduction across paired seeds and tasks would support a schedule explanation on fixed seven-shift data. Similar or higher OOD errors would count against that explanation. The experiment estimates the combined effect of the rate and warmup. It measures offline command-interface recovery.

## Intervention

| Setting | Original checkpoints | Candidate |
| --- | --- | --- |
| Peak learning rate | 0.001 | 0.0003 |
| Warmup | None | Linear from 0 through Optax count 200 |
| Decay | Cosine to 0.00005 | Cosine to 0.000015 |
| Updates, batch, seeds | 3,500, 64, 41–43 | Same |

Both schedules use a 3,500-count horizon. Optax consumes counts 0 through 3,499 during training. The nominal endpoint is count 3,500. The run records all three boundary rates and the last applied rate.

The driver calls the original Stage 1 Perceiver trainer once per seed with an optional schedule. Its default remains the original cosine. Every seed starts from `init_reader_v2(jax.random.key(seed), cfg)`. Source minibatches draw 64 indices with replacement using `np.random.default_rng(seed).integers(n_train, size=64)` in archive order. Training retains the original factor loss, global gradient clipping at 1, AdamW, and 0.0001 weight decay. The DIS/luminance reader has 32 positional slots, CNN width 96, 8 latents, Perceiver width 192, 3 layers, 4 heads and four output channels.

The original checkpoint run was `gpu03v2` at `d563845717e1f9312878b228328908484c4752f9`. Both arms use the scale campaign’s `gpu01/archive/archive.json`, 672 Panda and 672 Sawyer passive traces, 192 per shift. The driver verifies the ordered source tape hashes and retains the train/dev loader, DIS preprocessing, training sampler and end-of-seed dev audit. The 96 archive dev cases are used for that historical audit only. Training is passive-only and uses the Perceiver alone. [Stage 1 trainer](../xemb-icml-campaign-20261121-r1/stage1_offline_gate.py) · [PR7 schedule](../symmeta-wider-family-20261123-r1/eval_wider_family.py)

## Paired estimates

Score six exported checkpoint/seed combinations on identical `gpu09test/test_live` supports using the official loader. Verify the ordered digest of the 96 passive tapes and the three original checkpoint digests. The panel includes eight tasks with twelve shifts each. Four shifts per task are familiar, four compositional OOD, and four continuous-angle OOD. All predictors see three original five-control chunks. Each row pairs both predictions under the same seed, task, shift and tape hash.

The primary comparison is candidate minus original circular yaw MAE on the 64 OOD traces. Report overall, both OOD regimes, familiar, and pooled OOD. Retain yaw median, gain relative error, gripper sign accuracy, per-factor correctness and joint correctness under the historical Stage 1 rules (absolute circular yaw error ≤22.5°, relative gain error ≤0.25 and exact gripper sign). Planar operator error is the Frobenius norm of the residual corrected map divided by √2, using the PR7 implementation on gain times yaw rotation. Error reductions and accuracy increases favor the candidate. [Historical score](../xemb-icml-campaign-20261121-r1/stage1_offline_gate.py) · [Operator definition](../../src/fluffy_bassoon/xemb/wider_family.py)

Report per-seed results and the mean over the three fixed seeds. Draw eight whole tasks with replacement 20,000 times using seed 20261123. Keep both arms, all seeds and all included shifts together in each selected task. Report the 2.5th and 97.5th percentiles of the paired differences. Recompute each seed’s median yaw inside every draw before averaging those seed medians. The intervals quantify variation across the eight tasks conditional on these training seeds.

## Execution and records

Parent deployment and submission use one Baige Distributed Training RTX 4090 or RTX PRO 6000D. Expected cost is under one GPU-hour. The launcher takes an independently supplied full commit and a fresh attempt name. Check the device and first seed export after launch, then the run log every 20 minutes until completion. An exit-zero result contains three final Stage 1 msgpacks, three dev audits, three 96-row paired files, a machine-readable summary with all intervals, and a human-readable result. A nonzero exit preserves the attempt for diagnosis. Use a new attempt name for a retry. [Launcher](launch.sh) · [CPU qualification and launch command](READINESS.md)
