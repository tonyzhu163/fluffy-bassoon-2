# Trained paired continuation R4

Baige `job-zdic0jczdxoc` succeeded with 19,646 new source actions, 48 complete episodes and 12 complete K4 trials. Both original trained learners performed one PPO update and five Adam steps. The original 100-round source/evaluation goals remain unfinished.

| Learner | Source rounds | Source actions | Adam steps |
| --- | ---: | ---: | ---: |
| ppo | 56 | 536,686 | 280 |
| ppo_reset | 58 | 554,779 | 290 |

Allocated and independent final audits passed: full checkpoint SHA/ZIP CRC, native valid-mask action counts, all 12 likelihood/value replay errors exactly zero, durable group ACKs, config/Adam/sampler/RNG/evaluation recovery, canonical journal heads with zero pending or uncertain charges, frozen motor unchanged, child exits zero and clean owned process/GPU release.

Actual scheduler cost was 431 GPU-seconds; parent wall time 407.827073s is recorded separately. Unused 2269 operational GPU-seconds are sealed. This is source progress, with no new adaptation result or matched scalar speedup claim.

Runtime `57016ba08eed79043fd67bf8b085bd6d42e6d766`, launcher `c353970a316837b13277ec65587e58e5cb25ce56`; scientific lineage and original parent/canonical caps unchanged.

Receipts: [independent final audit](receipts/independent-final-audit.json), [allocated audit](receipts/audit.json), [terminal scheduler](receipts/scheduler-terminal.json), [reservation](reservation.json).
