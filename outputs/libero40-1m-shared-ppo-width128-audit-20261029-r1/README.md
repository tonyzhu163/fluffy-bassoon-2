# Width128 shared PPO bounded check

Baige `job-y48o6pc3928k`, source `3944ff79cc25b314dfffdbb7ce2fbf4d48d6fccf`,
run `libero40-1m-shared-ppo-width128-qual-20261029-r1`, exited0.
All221 focused allocated Linux checks passed; two JAX tests skipped in the
probe environment. Actual allocation GPU probe passed on NVIDIA RPBZZZ6.

Architecture width128/two layers/four heads/context300, learning rate1e-4,
two epochs, one K4 trial/update. Two source tasks Spatial00/01, eight fresh
episodes/eight successes,80 query chunks, two PPO iterations/four optimizer
steps.744 primitives+90 settling=834 physical actions within its4,000 action
reservation in the shared490k PPO pool. No unknown RPC intents remained.

Behavior likelihood replay error0; approximate KL.190/.0346, clip.359/.250,
finite actor/critic losses. Update windows18.61+29.53=48.14s for824 training
and settling actions; total launcher133.65s plus scheduler overhead.
The [time ledger](worker-time.json) charges194s including60s scheduler allowance.
This is neither all40 coverage nor evidence of stable learning, useful memory,
sample efficiency, robot improvement or target transfer.

[Independent saved-data audit](independent-artifact-audit.json) verifies all
eight raw trace/two behavior-policy hashes, explicit task/instruction and
current/next observations, actual latent/action/reward/duration/boundary
records, and full policy/optimizer/sampler/motor RNG/task-cycle recovery.
No model forward/backward or tests were run in that artifact inspection.
