# Shared PPO task-switch qualification

Baige `job-b7u5ogwy6udu`, `libero40-1m-shared-ppo-qual-20261029-r1`,
source `72bd91b988fa35acec324a6ea34cd2f427823120`, exited0. One policy,
optimizer and sampler retained across three fixed round-robin source goals:
Spatial00,01,02. Three K4 trials/12 episodes,12 successes,125 query chunks;
1,173 primitive+130 settling=1,303 physical actions. Three PPO iterations,
six optimizer steps.203 focused Linux tests passed/two probe-environment
JAX skips. Allocation GPU probe passed; hardware NVIDIA RPBZZZ6.

At learning rate1e-4/two epochs, behavior-log-prob replay error0 on each
update, finite losses, approximate KL.132/.026/.037 and clip fraction
.406/.296/.273. These are technical diagnostics on three easy tasks, not a
stable-learning comparison or memory/performance result. Width32/layer1/
four heads/context128 is an explicit small pilot, with one K4 trial/update;
the larger VLRLA batch is not reproduced.

The raw records bind each instruction and actual source-task identity.
Full recovery retains shared policy/optimizer, sampler/RNG, motor RNG and
task-cycle cursor/exposure. Only three of40 tasks actually have episodes;
the all40 schedule is not evidence of full collection. See [index](index.json)
and [metrics](metrics.jsonl). All costs count within the shared1M campaign.

[Independent saved-record inspection](independent-artifact-audit.json) verified
all12 raw traces and three behavior-policy hashes, actual executed latent
coordinates and the full learner/motor/task-cycle recovery. NumPy versus CUDA
tanh differs by at most1.1921e-7, within the collector's existing1e-6
unchanged-action tolerance. An initial audit-only1e-7 assertion was too strict;
this was no training failure or changed data/likelihood contract.
