# Complete RoboTwin PPO endpoints

Both PPO arms finished the original100 source rounds and all declared10/25/50/100 evaluation stages. Baige paired job-fji2xx8oq53h exited0 and reached scheduler/pod Success at2026-10-31 06:43:32 Shanghai. Final GPU memory was0, all176 owned-cleanup receipts were clean and no automatic failed retry occurred. This completes these two arms; the separate SAC arm remains in progress.

## Source endpoints

| Arm | Source rounds | Complete K4 trials | Source episodes | Actual source actions | PPO updates | Adam steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistent GTrXL/PPO | 100 | 600 | 2400 | 964875 | 100 | 500 |
| Independently trained episode-reset GTrXL/PPO | 100 | 600 | 2400 | 963759 | 100 | 500 |

Common frozen Motus pi0.5 H32/full1024 noise/bound1.5/features2062, seed20261030; global six completeK4 trials per update, five epochs, LR1e-6 and original native task horizons. Scientific lineage db0184089fa88a5f96c153f9b80029bb91b77def, migrated runtime57016ba08eed79043fd67bf8b085bd6d42e6d766, continuation supervisor0c16c2b794f8de5b78603206917e72195083c265. Sources and migration receipts are preserved; final evaluation adds no source actions or updates.

Source100 coordinator SHAs are9930fe4dbca5ff8cadaf8985e4f9cfe72c8feb050996e3d8f3f3ff7e558ebb52 (persistent slice042) and48e215501ef5e025a565855fbc240bd20bc9728e19046c7313703c74d837fb62 (reset slice040). Final coordinators arec439486c327116c8da4fc077217416757bdc2f21ffa819f8ae802cc3f7810fa5 (persistent slice043) and48d910980052ce2cec730669fbb6432b86d956aadbff22799b7ecefdeaadcb49 (reset slice041). Source, final and stage100 checkpoints independently passed boundSHA and263-memberZIP CRC checks. Adam/sampler/RNG/pending-rollout/stage/raw-corpus recovery is retained. Frozen motors and restored source RNGs are unchanged.

## Stage100 outcomes

| Controller/condition | Source validation | Bottle adjustment | Stapler placement |
| --- | ---: | ---: | ---: |
| Persistent normal | 2/24 | 7/8 | 1/8 |
| Same weights, memory reset | — | 6/8 | 1/8 |
| Same weights, reward blind | — | 8/8 | 1/8 |
| Separately trained episode reset | 3/24 | 8/8 | 0/8 |

Each target cell is one K8 trial, not eight independent trials. The72 learned-controller PPO episodes and40 trained-reset episodes match actual case seeds and initial feature/raw-observation hashes; all16 target cases are paired across conditions. Native before/after references are one episode per task/phase on a separate21161030 seed bank, each0/1. They are unpaired to these target cases and do not establish a native-vs-steered improvement.

| Controller/condition | Task | Success at episodes1/2/4/8 | All8 successes |
| --- | --- | --- | --- |
| ppo:normal | adjust_bottle | 1/1/1/1 | 11110111 |
| ppo:memory_reset | adjust_bottle | 1/1/1/0 | 11110110 |
| ppo:reward_blind | adjust_bottle | 1/1/1/1 | 11111111 |
| ppo:normal | move_stapler_pad | 0/0/0/0 | 00000010 |
| ppo:memory_reset | move_stapler_pad | 0/1/0/0 | 01000000 |
| ppo:reward_blind | move_stapler_pad | 0/0/0/0 | 00000010 |
| ppo_reset:normal | adjust_bottle | 1/1/1/1 | 11111111 |
| ppo_reset:normal | move_stapler_pad | 0/0/0/0 | 00000000 |

These results do not establish a persistent-memory advantage or an adaptation/generalization benefit. Reward-blind bottle success equals or exceeds normal; stapler success remains sparse. Source validation is low in both arms. One seed, two development tasks and one K8 trial per condition are insufficient for a broad method ranking. SAC and teacher/data qualification must be assessed separately.

Persistent final evaluation used25822 actions (76 episodes including four native references); reset used14639 (40 episodes). They add0 source actions/SGD. Closed original ledger charges are1068449 persistent and1021785 reset, both pending0/uncertain0 and under unchanged caps.

Actual paired scheduler duration was15451 GPU-seconds (4h17m31s); pod duration15455s is separate. The known terminal historical total becomes92784GPU-seconds=77333+15451, excluding the active SAC job and unverified630 historical optimization seconds. The13h cap was not actual usage or a reusable balance.

Evidence: [independent persistent audit](checks/independent-ppo100-20261031T0647.json), [independent reset audit](checks/independent-reset100-20261031T0632.json), [terminal accounting](receipts/scheduler-accounting-final.json), [target CSV](stage100-targets.csv). Audit SHAs: persistent aa6a3490334033b6b6250ed0ec28f5d1f023c059752920d6f4d828b85d9afe84; reset aea0ccc7e71f129b887625d98a22429ec70ac573a8fde35c4d483f67b3f58122.
