# Reset PPO100: source competence is weak; memory is not yet tested

Read-only diagnosis, 2026-10-31. No model/tensor loading, inference, tests, simulator work, allocations or live edits. `metadata-summary.json` binds the terminal summary, 100 update records, 2,400 episode records and four actual runtime source files by SHA. Source scientific pin is `db0184089fa88a5f96c153f9b80029bb91b77def`; final execution migration is `57016ba08eed79043fd67bf8b085bd6d42e6d766`. The reviewed local files in the RoboTwin process-pool checkout have exactly the same four file hashes as the actual executor.

## What the result supports

The reset actor completes 100 iterations, 600 K4 trials and 500 AdamW steps using 963,759 actual source actions. All 100 update logs and all 2,400 training episodes reconcile exactly, with no conflicting duplicate metadata. It succeeds **421/2,400 (17.54%)** during source training. Success in successive 25-iteration blocks is **112/600, 106/600, 102/600, 101/600**. This does not show a useful upward learning trend. The final source validation **3/24** is consistent with weak source training; it is not evidence of a dramatic generalization collapse from a strong teacher.

| Source goal | Training /400 | First25 /100 | Last25 /100 | Final VAL /4 | Horizon |
| --- | ---: | ---: | ---: | ---: | ---: |
| handover_mic | 76 | 20 | 17 | 0 | 600 |
| pick_diverse_bottles | 35 | 12 | 10 | 1 | 400 |
| place_a2b_left | 66 | 22 | 12 | 1 | 400 |
| place_object_stand | 138 | 31 | 34 | 1 | 400 |
| place_phone_stand | 86 | 24 | 19 | 0 | 400 |
| scan_object | 20 | 3 | 9 | 0 | 500 |

All 1,979 recorded failures reach the declared native task horizon. The cutoffs match the pinned benchmark task configuration, so failures are not explained by the old uniform400 cutoff truncating handover/scan. The horizon still defines the evaluated task; the logs do not show whether more time would recover failures. All training episode returns are exactly0 or1, matching sparse success feedback; reward magnitude is not silently scaled by1024 noise dimensions.

Validation is four fixed scene seeds per seen source goal: `3000000000 + task_index*10000 + episode_index*997`. Aggregate scores at iterations10/25/50/100 are **4/24,4/24,2/24,3/24**. The actor remains stochastic; fixed reset seeds alone do not imply the same sampled actions or byte-identical physical scenes across stages. Four scenes per goal give a coarse panel, with no matched native source baseline in this readout. It cannot establish regression versus the frozen native motor, broad source generalization, or why individual cases fail.

Target `adjust_bottle` **8/8** and `move_stapler_pad` **0/8** demonstrate strong task dependence. They do not establish that source training improved bottle success, since the corresponding native/full/SAC results are still needed. Eight different evaluation seeds are eight scenes, not eight attempts at one identical scene.

## What the reset control actually is

`vla_noise_ppo.py:161–185,237–252` clears recurrent cache and previous-noise/reward feedback at each reset for `reset_each_episode=True`. The full GTrXL still uses within-episode history. The episode-index clock remains in the public input (`:86–102`), clamped to the K4 training range at K8 evaluation. It is therefore a separately trained **episode-reset recurrent baseline**, not a fully memoryless MLP. No preceding episode outcome can reach its next episode through cache/previous feedback. Bottle8/8 is consequently not cross-episode adaptation evidence.

`robotwin_noise_compare.py:325–337` samples a shuffled block containing all six goals, with four independently sampled scene seeds for each K4 trial. A trial repeats the goal, not an identical physical scene. `:741–747` creates the reset policy with the actual reset flag; the generic `contract.ppo.reset_each_episode=False` describes the base config and is not proof that the reset arm used full memory. Actual episode receipts all record the reset behavior.

## Optimizer and noise: evidence versus hypotheses

The global update contains six complete trials/24 episodes. The actual batch is **265–348 valid chunk decisions, mean311.72**, totaling **31,172** decisions. `minibatch_trials=16` exceeds the six trials, so each of five epochs makes one optimizer step. Loss is correctly accumulated with valid-token weighting across whole recurrent trials (`vla_noise_ppo.py:305–364`). Longer failed tasks contribute more tokens; uniform goal scheduling is not uniform per-goal loss weighting.

All recorded losses are finite. Mean training approximate joint KL is **0.000232** (maximum0.001665), mean clip fraction **0.000163**, median clip fraction0. Reported pre-clipping gradient norms are **54.94–76.35**, all above the norm1 cap. These establish nonzero gradients and conservative policy movement; the reported KL is the weighted training metric, not a separately evaluated final-policy analytic KL. The log-probability is **summed across all1,024 noise coordinates** after the tanh Jacobian correction (`ppo.py:143–148`). Per-token log_ratio is joint new-minus-old log-probability; `approximate_kl = exp(log_ratio)-1-log_ratio` is averaged over valid chunk tokens and then token-weighted over the five training epochs (`vla_noise_ppo.py:329–359`). It is **not divided by1,024**: tiny values are not an artifact of reporting per-dimension KL. This sampled training estimate still is not an exact analytic final-policy KL. Actor/critic share AdamW LR1e-6, betas(.9,.95), decay.01, norm cap1. LR was previously selected for numerical stability, not demonstrated robot improvement. Lower LR, clipping, small batches or task interference could limit progress; none is isolated as the cause by these logs. Componentwise gradient norms and endpoint movement are not retained in the compact update logs.

The motor is frozen Motus pi0.5, three cameras, prefix mean2048 + proprio14, native action chunk32, five flow steps. PPO predicts **1024 independent noise coordinates (32×32)** using `1.5*tanh(N(mean,exp(clamp(log_std))))`; initial log_std=-.5 and the actor is not a zero-initialized native-preserving residual (`ppo.py:105–149`; `vla_noise_ppo.py:186–189`). This changes the initial-noise distribution; it is neither the published DSRL tied32D interface nor VLRLA's flow-SDE0.3 exploration. The native target control uses independent unbounded Gaussian initial noise (`robotwin_noise_compare.py:773–797`). A distribution mismatch is a concrete difference, not evidence it caused failure.

Reward is summed over actual executed chunk rewards, with first-success relative reward enabled and exact action masks (`robotwin_noise.py:363–371,623–661`). GAE uses gamma.99/lambda.95 **per chunk decision**, not per primitive action or duration. Only the final K4 boundary ends the return; episode terminal/truncation marks feedback/reset but GAE crosses earlier episode boundaries, including for the reset arm (`vla_noise_ppo.py:224–243,309–315`; `ppo.py:251–269`). This is intentional trial-return optimization, not four independent per-episode PPO objectives. There is no supported rollout/reward/GAE implementation defect from this review.

## Comparison with the already-recorded VLRLA reference

`docs/vlrla-protocol.md:89–136` records VLRLA source PPO as25 iterations,256 environments×4 rounds×8 nominal chunk slots=**8,192 slots/iteration**, five epochs, global minibatch2,048, actorLR5e-6/valueLR1e-4. Its source learner updates the VLA action expert; subsequent recurrent bridge fitting is supervised. Our learner directly optimizes a small recurrent noise controller over a frozen motor. Matching five epochs does not match what learns, batches or total exposure.

Our mean312 actual valid decisions/update is about **26.3× smaller** than that nominal8,192 capacity. Total31,172 actual decisions is **6.57× smaller** than VLRLA's204,800 nominal capacity;963,759 actual actions is **10.63× smaller** than its10,240,000 primitive-slot capacity. Published post-terminal valid counts are unavailable: these are actual-versus-capacity comparisons, not exact data ratios or a prescription to collect10M. More iterations100vs25 does not compensate for the batch difference. Shared LR1e-6 is5× below the reference actorLR and100× below its valueLR, but the optimized parameterizations and gradient scales differ, so transplanting the reference LR is not justified by the ratio alone.

Target novelty alone cannot explain the flat training history, and source-state generalization is not the sole observed weakness. The supported diagnosis is **weak/flat source learning with finite conservative updates**, plus a small fixed validation panel and strongly task-dependent target success. Full/reset/native/SAC robot comparisons remain essential. No new run, checkpoint choice, scope change or collection gate follows from this artifact.

## How to reproduce the metadata extraction

Run `extract_metadata.py` read-only on the development host; it imports only standard-library metadata utilities. It reads named original/continuation episode and update JSONLs, deduplicates inherited identities, refuses no data and reports conflicts/missing updates explicitly. It never interprets deduplicated metadata as reduced physical cost: the canonical final source receipt supplies963,759. The saved extraction finds100/100 updates,2,400episodes,31,172chunks and no conflicts. Pure actor tensors, replay buffers and raw frames remain unopened.
