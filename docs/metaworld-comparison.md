# Single-seed ML10 method comparison

The user selected one seed, a common 10-million-training-transition budget,
50 balanced training lanes, five 500-step episodes per trial, and ten PPO
epochs per collected batch. There are 125,000 new transitions per collection
and 80 collections. This follows METTA's interaction budget and its PPO epoch
count, while retaining explicit method adaptations described below. It does
not reproduce the unreleased METTA integration.

CTM uses T1 and cached 100-step recurrent replay, retaining full forward
history but truncating gradients. RL²-PPO uses the source-informed garage
GRU64/fitted-baseline adapter. ECET uses its hierarchical encoder, independent
actor/critic optimization and native transition minibatch size. AMAGO1 uses
the pinned official off-policy learner, B24/L2000, width256/layers3 and native
SDPA cache. AMAGO has no PPO epochs: the user selected ten sampled replay
passes' worth of transitions per new collection. Rational cumulative rounding
gives 2,083 optimizer updates and 99,984,000 sampled transitions over 10M new
transitions, versus a 100M target. This changes original AMAGO's update/data
schedule; sampled replay passes do not guarantee visitation of every stored
transition. Ten epochs do not equalize architecture cost or optimizer steps.

Use seed1000 and the same benchmark seed/pools, goal-visible state, physical
episode resets, per-episode task-instance resampling within a fixed family,
success latching and native reward records. Each policy receives one allocated
GPU and its own CPU/RAM resources. Record exact GPU identity; cross-paper
GPU-hour values from METTA's RTX3090Ti cannot be transferred to another GPU.

Evaluate fresh training-family instances and held-out families every five
collections. Use ten balanced lanes and 25 episodes, deterministic actions,
equal family/episode weighting, and the training public-clock encoding during
longer evaluation. Final paired conditions are normal, episode-memory reset
and reward blind. AMAGO's learned position table needs the explicit
evaluation-only extrapolation described in its method document. Keep raw
outcomes, family labels and returns. Report per-episode adaptation separately.

Plot measured fresh training-family success versus active allocated GPU-hours,
including CPU simulation and evaluation, plus companions versus training-only
GPU-hours and environment interactions. Do not extrapolate a capped run or
add uncertainty bands to one seed. Hardware, parameter counts, replay horizon,
method schedule and actual achieved steps must remain visible. A future
multi-seed result is separate work.

Before long training, qualify each method on its allocated GPU, verify frozen
behavior replay and complete optimizer/sampler/normalizer/RNG recovery, and
measure a real default-shaped update. AMAGO additionally needs its actual
B24/L2000 memory gate. A throughput-only checkpoint is not a learning run and
cannot silently resume into a different contract. Bound each allocation,
preserve complete checkpoints, and monitor advancing finite updates.

The serial collector is the reference. Optional subprocess simulation must
match seeds, observations, rewards, boundaries, task samples and raw outcomes
before enabling it. Charge process startup and communication. Profile
observation packing, policy/inference transfer and simulator stepping
separately; serial CPU stepping and recurrent GPU inference are distinct
contributors, not a policy-independent fixed cost.

Two additional CTM execution changes avoid unused all-false reset-state
construction and discarded synchronization normalization. Reference parity
must cover carried memory, outputs and parameter gradients. Compilation,
mixed precision and architectural width/depth changes remain unqualified;
none is assumed in runtime estimates.

References: [METTA Appendices D–E](https://arxiv.org/html/2610.31656v1#A4),
[CTM protocol](metaworld-ctm.md), [RL²-PPO](metaworld-rl2.md),
[ECET](metaworld-ecet.md), [AMAGO](metaworld-amago.md).
