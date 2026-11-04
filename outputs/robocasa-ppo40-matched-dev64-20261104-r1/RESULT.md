# Matched OpenDrawer DEV64

New PPO40 succeeded on 40/64 cases and GRPO40 on 31/64. New PPO succeeded on 13 cases where GRPO failed, and GRPO succeeded on four where new PPO failed. The prespecified GRPO-minus-new-PPO difference was −14.06 percentage points (descriptive paired 95% bootstrap interval −26.56 to −1.56, two-sided exact McNemar p=.04904). This is one trained checkpoint per method on a reused development panel.

| Checkpoint | First success | Success at the final control |
| --- | ---: | ---: |
| Human300 SFT | 36/64 (56.25%) | 35/64 (54.69%) |
| Old PPO40 | 33/64 (51.56%) | 33/64 (51.56%) |
| New PPO40 | 40/64 (62.50%) | 37/64 (57.81%) |
| GRPO40 | 31/64 (48.44%) | 28/64 (43.75%) |

| First-success comparison | Difference (pp) | Gains / losses | Paired 95% interval (pp) | Exact p |
| --- | ---: | ---: | ---: | ---: |
| Old PPO40 minus Human300 SFT | -4.69 | 3 / 6 | -14.06 to +4.69 | 0.5078 |
| New PPO40 minus Human300 SFT | +6.25 | 10 / 6 | -6.25 to +18.75 | 0.4545 |
| GRPO40 minus Human300 SFT | -7.81 | 3 / 8 | -17.19 to +1.56 | 0.2266 |
| New PPO40 minus Old PPO40 | +10.94 | 13 / 6 | -1.56 to +23.44 | 0.1671 |
| GRPO40 minus Old PPO40 | -3.12 | 3 / 5 | -12.50 to +4.69 | 0.7266 |
| GRPO40 minus New PPO40 | -14.06 | 4 / 13 | -26.56 to -1.56 | 0.0490 |

The new-PPO versus SFT difference changed sign across the original and matched evaluations. In the original periodic run, new PPO went from 34/64 to 26/64 (−12.5 pp, p=.0963). Here it scored 40/64 against SFT’s 36/64 (+6.25 pp, p=.4545). Both estimates remain uncertain. The original decline did not reproduce on this common draw. The other pairwise contrasts are descriptive. The separate PPO ablation owns the fresh 512-case readout.

All 33 eight-case batches matched the original physical state, public state and task description. Case membership and all 396 Torch CPU/CUDA sampler records verified. Across comparisons to the reference resets, the largest initial pixel difference was one intensity level. Repeating SFT on the first eight cases with the same noise gave the identical first action chunk, then one first-success and end-success outcome changed (seed 610102).

The run used 158,400 valid policy controls, 158,400 journaled controls and zero training updates. All 667 inference actor tensors verified for the converted checkpoints. The two allocations used 0.693 scheduler GPU-hours combined. The first attempt ended before model inference because the CUDA visibility selector was missing.

[Paired results and end-success contrasts](RESULT.json) · [independent episode and RNG verification](DEV64-VERIFICATION.json) · [model hashes](READY-FOUR.json) · [execution and costs](EXECUTION.json).
