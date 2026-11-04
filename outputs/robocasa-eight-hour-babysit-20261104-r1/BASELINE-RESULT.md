# Fresh independent PPO40–GRPO40 TEST512

The prespecified primary comparison establishes no winner: GRPO322/512(62.89%) versus PPO303/512(59.18%), paired+3.71094percentage points,83wins/64losses, two-sided exact McNemar p=.137378, paired95% bootstrap interval[-.78125,+8.39844]points. SFT is311/512(60.74%). Descriptive PPO−SFT is−1.5625points(p=.5543); GRPO−SFT is+2.14844points(p=.2945). The earlier matched DEV64 advantage for PPO over GRPO did not repeat on this independent panel. This is one trained checkpoint/seed per method and does not establish equivalence or a general method ranking.

Both learned actors used3,072,000 TRAIN robot controls and their fixed policy40 endpoints. Evaluation used the same512 fresh physical cases, fixed B8 cohorts, common sampler RNG and counterbalanced policy order. Only the actor's original667 GRPO tensors were used; eight unused value-head tensors supplied the common inference architecture, with exact actor-byte preservation and no SGD. Final-step successes were SFT299/PPO290/GRPO309; the prespecified primary outcome remains first success.

Independent allocated-Linux CPU auditing verified all1544 complete episodes (including eight repeated SFT controls),1544 matched physical/public starts,2316 common-RNG sampler chunk calls,926400 resolved physical controls, all endpoint/canonical hashes and exact primary discordance/p-value arithmetic. Both initial-policy likelihood checks have zero log-ratio error. The SFT repeat retained all eight outcomes, but only its first action chunk was byte-identical; later action differences remain recorded, so exact trajectory determinism is not claimed.

The separate four-epoch PPO experiment remains positive:355vs315/512 over the one-epoch PPO, paired+7.8125points, Holm-adjusted p=.0007748. That is a different paired case bank; this fresh panel contains no four-epoch GRPO or four-epoch PPO, and cannot directly rank those endpoints against one another. The two new-source proposal therefore uses the supported four-epoch PPO change while keeping its cross-task effectiveness conditional on fresh tests.

Baige job job-ynq6hyyggfj6 succeeded and released01:32:58 Shanghai. Conservative creation-to-release accounting is12432GPU-seconds(3.45333GPU-hours), below its separate7.5GPU-hour all-attempt ceiling. Zero evaluation SGD; no retry or extension.

[Result](BASELINE-RESULT.json) · [independent audit](BASELINE-INDEPENDENT-AUDIT.json) · [manifest](BASELINE-MANIFEST.json) · [execution](BASELINE-EXECUTION.json). Raw referenced artifacts remain on project storage.
