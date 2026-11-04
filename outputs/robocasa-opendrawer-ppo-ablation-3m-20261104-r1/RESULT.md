# Four epochs improved OpenDrawer PPO at 3,072,000 training actions

Four passes over each fresh rollout improved held-out first success from 315/512 to 355/512 with the Stage 1 prefix critic and denoising rule. The paired gain was **7.81 percentage points** (95% interval +3.71 to +11.91). The data support four epochs for this OpenDrawer recipe at this training budget. The historical PPO40 checkpoint and Stage 1 were within one percentage point of each other on the common test.

## Six policies on the same 512 fresh cases

The primary metric is at least one native success within 600 simulator controls. Wins count cases solved by the variant alone. Losses count cases solved by Stage 1 alone. Intervals are descriptive paired case-bootstrap 95% intervals. The three primary two-sided exact McNemar tests use Holm adjustment.

| Policy | Successes | Rate | Change vs Stage 1 | Wins / losses | 95% interval, pp | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Starting SFT | 303/512 | 59.18% | −2.34 pp | 49 / 61 | — | — |
| Historical PPO40 | 320/512 | 62.50% | +0.98 pp | 78 / 73 | −3.91 to +5.66 | Reference |
| Stage 1 PPO40 | 315/512 | 61.52% | Reference | — | — | — |
| Four epochs only | 355/512 | 69.34% | **+7.81 pp** | **78 / 38** | **+3.71 to +11.91** | **0.000775** |
| Sampled-step suffix critic only | 330/512 | 64.45% | +2.93 pp | 58 / 43 | −0.98 to +6.84 | 0.327 |
| Last denoising step eligible only | 330/512 | 64.45% | +2.93 pp | 65 / 50 | −1.17 to +7.03 | 0.327 |

The historical-versus-Stage-1 descriptive paired p was 0.745. The current comparison estimates each change with the other two settings held at Stage 1. The suffix treatment includes its native head dimensions and initialization. All 667 non-head starting tensors matched. These results cover one training seed and one task. Interactions among changes and variation across training seeds remain open. [Machine-readable results](ENDPOINT-RESULT.json) · [Outcomes by case](TEST512-CASES.jsonl) · [Verification from native episode receipts](ENDPOINT-AUDIT.json) · [Predeclared design](PLAN.md)

## Training and diagnosis

Every PPO40 policy in the table received 40 fresh rollouts of 128 episodes × 600 controls, totaling 5,120 episodes, **3,072,000 executed training actions**, or 61,440 policy decisions issuing 50-action chunks. Historical PPO and the four-epoch arm each took 960 Adam steps per actor rank. Stage 1 and the other two variants took 240. Matching 240 optimizer steps instead would select historical update 10 at 768,000 actions. The historical +7.42-point held-out improvement was measured at update 130 and 9,984,000 training actions. [Source comparison](../robocasa-ppo-grpo-meta-stage1-20261104-r1/ROOT-CAUSE-ANALYSIS.md) · [Historical 512-case endpoint](../robocasa-endpoint512-acceleration-20261104-r1/RESULT.md)

The historical implementation stored the mean suffix value over denoising steps while learner replay evaluated the selected step. The corrected Stage 1 runtime and all three new arms passed their stored-versus-replayed likelihood/value checks. Each new run passed 80 rank/update checks, retained its frozen VLM and finished exactly 40 updates. The logged mean clipping fraction was 4.80% for Stage 1 and 17.46% for four epochs. The final 64-case development scores were 26 for Stage 1, 38 for four epochs, 33 for the suffix critic and 35 for last-step eligibility. The two historical headline DEV scores used different banks. [Training audit E](E-COMPLETE.json) · [C](C-COMPLETE.json) · [D](D-COMPLETE.json) · [Paired development panels](DEV-THROUGH40.json)

One initial attempt stopped before any optimizer step because its verification compared asynchronous actor-rank positions. Joining the saved rollouts by unique initial noise recovered all 128 case identities, with identical noise and selected indices. The corrected check was qualified on those actual batches before relaunch. Each arm subsequently collected trajectories under its own policy. Async receiving-rank order and small rendering differences can change trajectories or minibatch membership. [First-collection verification](FIRST-COLLECTION-R2.py) · [Initialization](PREFIX-START.json) · [Suffix initialization](SUFFIX-START.json)

## Test and cost

Four isolated Baige workers evaluated 64 fixed eight-case banks, each with the six frozen policies in counterbalanced order. All starts matched in physical and public state and all 12 sampler RNG transitions matched within each bank. The repeat of the initial policy on bank zero produced the same eight success/failure outcomes. Its first action chunk was identical. Its subsequent continuous action sequences diverged. The audit covers 3,080 complete policy episodes and 1,848,000 evaluation controls, separate from training. [Bound policy hashes and case order](EVAL-MANIFEST.json) · [Independent audit](ENDPOINT-AUDIT.json)

Baige reported **64.79 GPU-hours** across the three training arms, the stopped first attempt and the shared test. Charging from creation to release gives a conservative **65.01 GPU-hours**, within the separate approved 92. All allocations were released. [Budget closeout](BUDGET-CLOSEOUT.json)
