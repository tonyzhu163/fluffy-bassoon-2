# Frozen DIT paired confirmation

256 fresh paired cases per task; 11,520 audited physical episodes. Endpoint: success during episode six.

| Panel | DIT full | DIT reset | SFT | PPO130 |
| --- | ---: | ---: | ---: | ---: |
| ID | 43.2% | 44.9% | 41.1% | 49.7% |
| OOD | 46.1% | 44.1% | 43.4% | 49.4% |

| Panel | DIT contrast | Difference (pp) | Paired 95% interval (pp) | Wins / losses |
| --- | --- | ---: | --- | ---: |
| ID | Full − DIT_episode_reset | -1.69 | [-4.43, 1.17] | 54 / 67 |
| ID | Full − original_SFT | +2.08 | [-1.17, 5.34] | 91 / 75 |
| ID | Full − fixed_PPO130 | -6.51 | [-10.29, -2.60] | 91 / 141 |
| OOD | Full − DIT_episode_reset | +1.95 | [-1.17, 5.08] | 39 / 29 |
| OOD | Full − original_SFT | +2.73 | [-1.17, 6.64] | 61 / 47 |
| OOD | Full − fixed_PPO130 | -3.32 | [-7.23, 0.59] | 45 / 62 |

Task-stratified paired case bootstrap, equal task weighting; marginal95% intervals without multiplicity correction. Conditional on one fixed DIT fit and these five tasks, not training-seed or task-population uncertainty.

History benefit and advantage over SFT/PPO support separate claims. These results do not explain why DIT differs, establish seed robustness, or isolate task diversity. OOD is relative to the meta-training corpus. No continue/kill numerical threshold was predeclared.
