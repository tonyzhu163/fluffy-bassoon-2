# Recipe candidates implemented and CPU qualified; no GPU run launched

Three named candidates preserve the completed Stage1 run identities:

| Candidate | TRAIN controls | Adam steps/rank | Executed / predicted controls | K / noise / global minibatch |
| --- | ---: | ---: | --- | --- |
| `ppo_reuse4` | 3,072,000 | 960 | 50 / 50 | 10 / .3 / 256 |
| `ppo_long` | 12,288,000 | 960 | 50 / 50 | 10 / .3 / 256 |
| `ppo_short_chunk` | 3,072,000 | 600 | 10 / 50 | 5 / .5 / 512 |

All defaults use zero critic warmup. The native optimizer is unchanged; the
untested optimizer-rewrite draft was removed. The old primary PPO config is
semantically identical after moving its shared body into a reusable config.
Only primary configs set Hydra's search path. Two failed composition checks
were diagnosed before any model or GPU construction; the successful R3
checks all three candidates and the unchanged original Stage1 GRPO contract.

**61 Linux CPU checks pass** in214.34seconds. Subsequent changes affect only
configuration composition; the tested learner/test source content is unchanged.
The pinned actual native sampler method was also checked with synthetic
features at all four allowed K5 stochastic indices. It retains the50-control
latent horizon, executes10 controls and computes likelihoods for120 executed
coordinates. Independent Gaussian-density reduction has maximum chunk
log-ratio error7.63e−6, below1e−3; VLM-prefix value error is zero.
These checks construct no transformer and do not qualify actual GPU inference.

[CPU qualification](CPU-QUALIFICATION.json),
[resolved contracts](RESOLVED-CONTRACTS.json),
[sampler qualification](SAMPLER-CPU-QUALIFICATION.json),
[plan and limits](PLAN.md).

Current pending implementation source is00b25d1b120a0a155230cb43b4adec54a4ccffef,
deployed clean by Git bundle at
`/mowenjie/fluffy-bassoon-2/sources/recipe-fixes-00b25d1b`.
GitHub publication remains blocked by authentication; do not report it pushed.
No changed-recipe GPU jobs have been submitted. Actual-device timing, memory,
native episode and real-weight likelihood qualification must precede a sized
training proposal. A warmup treatment would be a separate explicit ablation.
