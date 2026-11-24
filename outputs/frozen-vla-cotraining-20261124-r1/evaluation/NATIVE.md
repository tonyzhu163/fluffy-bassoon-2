# Locked native transfer and Panda retention

The jointly trained policy succeeded on **7/12 Kinova3 starts**, compared with **8/12 for stock**. It lost a stock success on drawer row 46 and rescued none. On **Panda it succeeded on 6/6**, retaining all five stock successes and rescuing cheese row 49. Removing the one discordant start for either arm eliminates its observed joint–stock difference. These outcomes support history-dependent behavior and a retention benefit on these six Panda starts. They give no evidence of an aggregate transfer gain on the held-out Kinova3 panel.

## Native outcomes

| Condition | Kinova3 successes / 12 | Panda successes / 6 |
| --- | ---: | ---: |
| Stock | 8 | 5 |
| Policy only | 8 | 5 |
| Adapter only | 9 | 4 |
| Joint | 7 | 6 |
| Joint, zero | 6 | 4 |
| Joint, lagged | 9 | 4 |

On the six released layouts shared by both arms, joint and stock each succeeded on the same 3/6 Kinova3 starts. The Panda joint–stock difference on those layouts was one success. Counting all 12 Kinova3 starts, every condition completed all four cheese starts. Drawer rows 47 and 48 timed out under every condition. On moka pots, adapter-only retained both stock successes and added row 49. Policy-only also scored 3/4, adding rows 48 and 49 while losing row 47. Joint and stock each completed the same 2/4.

A rescue is a start on which joint succeeded and the named comparator failed. A loss is the reverse. Differences below are joint minus comparator in percentage points. The paired 95% percentile intervals use 20,000 resamples of starts within each of the three fixed tasks, seed 20261124. Each resampled start keeps its six outcomes together.

| Joint compared with | Kinova3 rescues / losses | Kinova3 difference, pp [95%] | Panda rescues / losses | Panda difference, pp [95%] |
| --- | ---: | ---: | ---: | ---: |
| Stock | 0 / 1 | -8.3 [-25.0 to 0.0] | 1 / 0 | +16.7 [0.0 to +33.3] |
| Policy only | 1 / 2 | -8.3 [-33.3 to +16.7] | 1 / 0 | +16.7 [0.0 to +33.3] |
| Adapter only | 0 / 2 | -16.7 [-41.7 to 0.0] | 2 / 0 | +33.3 [0.0 to +66.7] |
| Joint, zero | 1 / 0 | +8.3 [0.0 to +25.0] | 2 / 0 | +33.3 [0.0 to +66.7] |
| Joint, lagged | 0 / 2 | -16.7 [-41.7 to 0.0] | 2 / 0 | +33.3 [0.0 to +66.7] |

Joint minus policy-only minus adapter-only plus stock was −16.7 points on Kinova3 (95% interval −50.0 to +16.7) and +33.3 on Panda (0.0 to +66.7). Every displayed contrast interval includes zero. The intervals describe variation among these starts for one fitted checkpoint set, one noise bank per start and three chosen tasks. They do not quantify uncertainty across training replicates or a population of tasks.

## What changing history did

With fixed joint weights, supplying aligned history rescued Kinova3 moka-pot row 47 relative to zero history. Delaying history by five controls preserved that success and also completed Kinova3 drawer 46 and moka pots 48, both failures under aligned history. On Panda, aligned history rescued cheese 49 against both controls, moka pots 48 against zero and moka pots 49 against lagged. Thus the two Panda control totals of 4/6 conceal different failed starts. The benefit of access to history on one Kinova3 case does not establish a benefit from its exact timing.

On **all 18 starts**, joint first differed from stock, policy-only and adapter-only in both its proposal and issued command at control 0. Joint first differed from each same-weight history control at control **5**, the first replan with executed history. At control 10, shadow calls changed only the supplied history while holding the joint trajectory's observation, weights and diffusion noise fixed. The table gives the median across starts of the largest absolute coordinate change among the first five decoded commands, in normalized controller units. All 18 aligned-versus-lagged comparisons changed a proposal.

| Same-input comparison at control 10 | Kinova3 median | Panda median |
| --- | ---: | ---: |
| Aligned vs deployed zero history | 0.005893 | 0.004750 |
| Aligned vs five-control lag | 0.001754 | 0.001416 |
| Aligned vs zero offset through the same history computation | 0.005692 | 0.005051 |
| Zero offset through history computation vs native zero route | 0.001430 | 0.001156 |

The learned offset changed the proposal on the same computation path on all 18 starts, by more than the measured zero-route discrepancy on each start. The terminal comparisons estimate the effect of the deployed history interventions, including their numerical routes and all subsequent closed-loop state changes. The same-input checks isolate an action effect. They cannot assign a terminal rescue to a particular history field or determine how much of it came from the numerical route.

## Each registered start

A number is the executed control count at the single terminal reward of 1. “Fail” means every native reward was zero through the full 300 controls for drawer and cheese, or 520 for moka pots. The six conditions produced 75 terminal successes and 33 full-horizon failures. The machine-readable record also gives the first differing proposal and issued control for each of the 90 paired joint comparisons. [Per-start outcomes, divergences and original file hashes](evidence/native-results.json)

| Arm | Task | Row | Stock | Policy only | Adapter only | Joint | Joint, zero | Joint, lagged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kinova3 | Drawer | 46 | 184 | fail | 182 | fail | fail | 240 |
| Kinova3 | Drawer | 47 | fail | fail | fail | fail | fail | fail |
| Kinova3 | Drawer | 48 | fail | fail | fail | fail | fail | fail |
| Kinova3 | Drawer | 49 | 179 | 219 | 234 | 253 | 225 | 226 |
| Kinova3 | Cheese | 46 | 100 | 99 | 99 | 99 | 99 | 99 |
| Kinova3 | Cheese | 47 | 99 | 100 | 106 | 105 | 106 | 105 |
| Kinova3 | Cheese | 48 | 105 | 105 | 110 | 105 | 106 | 105 |
| Kinova3 | Cheese | 49 | 119 | 116 | 116 | 116 | 116 | 116 |
| Kinova3 | Moka pots | 46 | 449 | 398 | 456 | 479 | 451 | 420 |
| Kinova3 | Moka pots | 47 | 441 | fail | 370 | 403 | fail | 401 |
| Kinova3 | Moka pots | 48 | fail | 426 | fail | fail | fail | 424 |
| Kinova3 | Moka pots | 49 | fail | 446 | 357 | fail | fail | fail |
| Panda | Drawer | 48 | 186 | 250 | 185 | 190 | 245 | 186 |
| Panda | Drawer | 49 | 210 | 239 | 205 | 192 | 205 | 210 |
| Panda | Cheese | 48 | 95 | 83 | 92 | 93 | 95 | 95 |
| Panda | Cheese | 49 | fail | 99 | fail | 248 | fail | fail |
| Panda | Moka pots | 48 | 431 | 414 | 423 | 439 | fail | 400 |
| Panda | Moka pots | 49 | 515 | fail | fail | 393 | 377 | fail |

## What was held fixed

The three task identities were chosen from prior native work. Rows 46–49 were registered on Kinova3 and rows 48–49 on Panda before fitting or opening this panel. All three tasks were absent from the additional offline training and validation data. Stock π₀.₅ has its own public LIBERO pretraining exposure. Policy-only, adapter-only and joint used the same offline examples and training schedule. At evaluation, the stock checkpoint, output projections, history weights and normalization stayed frozen. The four-transition buffer was reset at every start and updated from issued commands and measured tool and jaw motion. Zero and lagged controls used the same joint weights, with either an empty buffer at every replan or a buffer delayed by five controls. [Preregistration](PANEL.md) · [Source split](../data/RESULT.md) · [Training interface](../model/API.md)

For each arm and start, all six conditions had identical initial physical state, camera views, proprioception, transformed policy inputs, instruction and complete noise bank. Matched Panda and Kinova3 rows shared the released scene and row, prompt, noise bank, action schema and horizon. The stock and no-op action chunks matched bit for bit on the historical check and every scored opening. The initial stock and adapter-only actions matched, as did all three joint opening actions. A trained projection was free to change its opening proposal.

The independent CPU readback reconstructed the causal buffer and per-replan noise, checked each clipped action against the issued prefix, re-summed every raw native reward and required each failure to reach the released horizon. It verified the exact expected set of arm/task/row/condition files, regenerated the GPU job's complete scored summary, and checked the frozen model and fit against the trainer's effective parameter hashes. [Verifier](verify.py) · [Independent reward readback and raw hash index](native_readback.py)

## Deployment and provenance

Baige Distributed Training job `job-tda9w2zu35vq` ran on one RTX 4090 at evaluation revision `7c7d67423e8b874420a38867ee5ee9cc389c0e0c`, from 12:09:02 to 12:34:03 Shanghai on November 24, 2026. It exited 0. All 108 resets matched on the first attempt. The CPU readback verified **30,189 native control rewards** across the exact 18 × 6 episodes. The before and after files were byte-identical, SHA-256 `24557d1397635f49a9c6508993bc60197ed679aeb27b97f82212702955dc13e9`. This covers resident stock weights, each fitted head and effective kernel, normalization and the on-disk fit. The trainer's `model/gpu02/fit.json` SHA-256 was `4855546591381f149d7509a6af3d45db32e9e82a732788c8376d504f19417dbc`. Evaluation used forward inference throughout.

The original job `job-ko1foqhzyc7h` stopped at its historical Panda parity check before opening any registered layout. Forcing a zero offset through the compiled history operation changed an executable-prefix coordinate by 0.000957 at identical input and noise. The recovery dispatches an exactly zero fitted offset through the native sampler with the condition's materialized projection. The deployed zero-update head with valid history then reproduced unmodified stock exactly. The repair preceded every held-out outcome. The two attempts consumed 193 and 1,501 job-step seconds respectively, **0.471 GPU-hours** total. [Failed parity check and artifact hashes](evidence/native-attempt01.json) · [CPU dispatch check](evidence/native-dispatch-fixture.json) · [Recovered sampler](deployed.py)

Original files are on Baige PFS under `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1/evaluation/gpu02`. The compact readback contains SHA-256 digests for the 108 raw episode files and ten runtime records, including the frozen parameter hashes, device probe, parity check, exit and scored summary. Its own SHA-256, checked on Baige and after transfer, is `d88c6be95549f889cb1db46dfa66dd0d7567aeb8c337c1d08e2a825be82511d1`. [Verified readback](evidence/native-results.json)
