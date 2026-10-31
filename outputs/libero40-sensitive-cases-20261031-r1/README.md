# Which observed sensitive cases did fixed9 omit?

2026-10-31. Saved-result metadata only; no model/trace execution, new rollout,
tests, GPU work or launch. Group index hashes were checked while extracting
actual K0 outcomes. Every comparison below uses **its own wave's native** and
matching query seed, nominal state and actual initial-observation hashes.
Native31/90,30/90 and29/90 are not pooled or substituted across waves.

|Wave / method |Own native →method score |Native-only success IDs |Method-only success IDs |Discordant /omitted by fixed9 |
| --- | --- | --- | --- | ---: |
|Student AD full |31→26 /90 |44,59,70,77,79,82 |38 |7 /6 |
|Student AD static |31→26 /90 |12,33,44,59,77,79,82 |29,38 |9 /9 |
|Student DPT full |31→26 /90 |12,33,44,59,70,77,79 |29,38 |9 /8 |
|Student DPT static |31→27 /90 |12,44,70,77,79 |38 |6 /5 |
|PPO full |30→24 /90 |9,20,31,38,54,77,79,82 |26,33 |10 /9 |
|PPO trained episode reset |30→26 /90 |2,9,28,31,54,70,77 |44,59,74 |10 /9 |
|Noise zero |29→25 /90 |2,59,77,79 |none |4 /4 |
|Noise tied Gaussian |29→28 /90 |30,38,64,77,79 |14,31,43,82 |9 /8 |
|Noise tied clipped |29→29 /90 |22,30,77 |43,74,82 |6 /5 |

Each case ID means `libero90:ID/state-0/seed-123`. These are **observed
single-case outcome switches**, not stable labels of controllability.

- Student-wave union:10 sensitive IDs; **9 omitted** by fixed9.
- PPO-wave union:16 sensitive IDs; **14 omitted**.
- Noise-control union:13 sensitive IDs; **12 omitted**.
- Union of IDs across the separate comparisons:23; **20 omitted**:
  **2,9,12,14,22,26,28,29,31,33,38,43,44,54,59,64,74,77,79,82**.
- Fixed9 contains observed sensitivities20,30,70. Its other six IDs are
  0,10,40,50,60,80. No score or statistical trial count is formed by pooling
  the three waves; the union is only a development-selection map.

The now-completed distribution panel gave all seven methods6/9, all failing
0/40/80. It missed most previously observed switches, but was **not guaranteed
insensitive a priori**: student70, PPO20/70 and noise30 switched earlier.
Equality on this panel/run does not establish universal mean/std invariance.
Any future panel chosen using this map must be labeled outcome-selected
**development diagnosis**, while the all40 source/full90 target protocol and
all original benchmark cases remain unchanged.

## Task0 checkpoint80 versus160: paired condition, different learner state

|Frozen checkpoint |Optimizer updates |Query seed |Success |Primitive actions |
| --- | ---: | ---: | --- | ---: |
|After80 training episodes |54,140 |238430264 |yes |249 |
|After160 training episodes |104,860 |238430264 |no |400 |

The nominal state hash and actual initial-observation hash match exactly.
Full learner signatures and immutable milestone recovery hashes differ
(details in JSON). The signature includes nonactor state, so its difference
alone is not an isolated actor-parameter test. The pinned query implementation
resets actor and motor PRNG keys from the recorded seed and restores training
RNG after the query. Generated noise/actions can change because the restored
SAC state differs; paired seed numbers do not make them identical.

This is **not a repeated measurement of the80-episode checkpoint**. One query
per changed checkpoint does not estimate either success probability or prove
a true performance-rate regression. The38 successful training episodes
reported by the parent at160 also do not make a fixed query seed's success
certain. No changing learner checkpoint was opened or model evaluated here;
only the already committed80/160 query receipt entries were inspected.

## One concrete next development proposal—unlaunched

Freeze the exact80- and160-episode task0 milestone recoveries and compare both
with contemporaneous native using **eight prespecified stochastic query seeds
1000…1007**, preserving physical state0/scene seed123. Use one common motor
process, matched actual initial arrays, zero SGD/replay insertion, and paired
actor/motor RNG across conditions. The query-seed override must be separate
from the physical scene seed.

Three conditions ×eight repeats =24 query episodes; worst9,840 executed
actions including400+10 per episode, proposed10,500-action ceiling. Root must
first reconcile an explicit compute/action budget after currently funded runs;
this proposal creates no reservation or launch. Report per-seed discordances
and censored lengths. These are eight stochastic repeats of **one**
initialization, not eight tasks, and the task/checkpoints are outcome-selected.
Do not choose80 as a new best checkpoint or report this as confirmatory
meta-learning, generalization or an adaptation sample-efficiency result.

`receipt.json` preserves all nine comparison case/outcome lists and trace/group
hash references, wave source pins, omitted-ID counts, and the two immutable
query receipt/checkpoint hash references. It is a sensitivity map, not a new
semantic filter or source-data admission rule.
