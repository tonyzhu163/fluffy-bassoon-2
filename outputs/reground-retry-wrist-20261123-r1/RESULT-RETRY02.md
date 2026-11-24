# Native wrist intervention on separately calibrated Panda retries

The pinned Baige Distributed Training attempts sealed all 70 batches. The independent CPU readback verified every batch. The 280 fixed task–starts each had 15 passive support controls on a separate layout and two fresh scored attempts. Both retries used the same frozen estimate and native opening proposal. The added intervention rotates only angular XY by inverse estimated yaw. [Protocol](PLAN.md) · [Launch and first-pair audit](RETRY02.md)

Adding wrist correction changed native success by +17.50 [+13.57, +21.43] percentage points and unsafe incidence by -10.71 [-15.00, -6.43] points. Brackets give the registered 95% paired-start intervals.

| Panel | Native outcome | Translation | Translation+wrist | Historical streaming, label 0 | Wrist-only / translation-only | Wrist − translation, pp [95%] | Support-cluster 95% interval, pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | success | 181/280 | 230/280 | 81/280 | 60/11 | +17.50 [+13.57, +21.43] | [+13.03, +21.99] |
| all | unsafe | 121/280 | 91/280 | 75/280 | 13/43 | -10.71 [-15.00, -6.43] | [-15.32, -6.19] |
| goal_1 | success | 35/35 | 34/35 | 35/35 | 0/1 | -2.86 [-8.57, +0.00] | [-8.57, +0.00] |
| goal_1 | unsafe | 1/35 | 3/35 | 0/35 | 2/0 | +5.71 [+0.00, +14.29] | [+0.00, +14.29] |
| goal_2 | success | 0/35 | 32/35 | 0/35 | 32/0 | +91.43 [+82.86, +100.00] | [+81.82, +100.00] |
| goal_2 | unsafe | 22/35 | 0/35 | 8/35 | 0/22 | -62.86 [-77.14, -45.71] | [-75.76, -51.28] |
| 10_1 | success | 10/35 | 14/35 | 3/35 | 4/0 | +11.43 [+2.86, +22.86] | [+0.00, +25.00] |
| 10_1 | unsafe | 29/35 | 31/35 | 33/35 | 3/1 | +5.71 [-5.71, +17.14] | [-6.06, +20.59] |
| 10_3 | success | 7/35 | 21/35 | 2/35 | 16/2 | +40.00 [+20.00, +60.00] | [+20.00, +60.61] |
| 10_3 | unsafe | 9/35 | 5/35 | 0/35 | 2/6 | -11.43 [-25.71, +2.86] | [-30.30, +5.26] |
| object_1 | success | 30/35 | 26/35 | 18/35 | 2/6 | -11.43 [-25.71, +2.86] | [-26.47, +2.86] |
| object_1 | unsafe | 1/35 | 3/35 | 1/35 | 2/0 | +5.71 [+0.00, +14.29] | [+0.00, +14.71] |
| object_3 | success | 29/35 | 33/35 | 4/35 | 6/2 | +11.43 [-2.86, +25.71] | [-6.25, +31.58] |
| object_3 | unsafe | 16/35 | 12/35 | 14/35 | 2/6 | -11.43 [-25.71, +2.86] | [-27.78, +5.56] |
| spatial_0 | success | 35/35 | 35/35 | 16/35 | 0/0 | +0.00 [+0.00, +0.00] | [+0.00, +0.00] |
| spatial_0 | unsafe | 33/35 | 35/35 | 11/35 | 2/0 | +5.71 [+0.00, +14.29] | [+0.00, +14.71] |
| spatial_2 | success | 35/35 | 35/35 | 3/35 | 0/0 | +0.00 [+0.00, +0.00] | [+0.00, +0.00] |
| spatial_2 | unsafe | 10/35 | 2/35 | 8/35 | 0/8 | -22.86 [-37.14, -8.57] | [-38.89, -8.82] |

The primary intervals resample paired starts within each of the eight fixed tasks. The sensitivity resamples support layouts within task. The five-label historical comparison averages the five labels within each task–start. The fixed task mix, policy, reader and support assignment define these estimates.

For batch054 on candidate 2, the first completed support and translation rollout came from the earlier launch. An independent audit verified the preserved proposal and matched every recorded input leaf to the admitted wrist reset. The wrist rollout used the first eligible reset and ran once to the native endpoint. The result is conditional on exact renderer matching. [Sequential rule and budget](recovery/sequential-reset/RETRY07.md)

## Historical comparisons

| Native outcome | All-task contrast | Difference, pp [95%] | Support-cluster 95% interval, pp |
| --- | --- | ---: | ---: |
| success | translation − stream0 | +35.71 [+30.71, +40.71] | [+29.90, +41.53] |
| success | translation − stream5 | +36.00 [+31.00, +41.00] | [+30.18, +41.85] |
| success | translation+wrist − stream0 | +53.21 [+48.21, +58.21] | [+47.13, +59.24] |
| success | translation+wrist − stream5 | +53.50 [+48.50, +58.57] | [+47.46, +59.56] |
| unsafe | translation − stream0 | +16.43 [+11.07, +22.14] | [+10.19, +22.68] |
| unsafe | translation − stream5 | +16.43 [+11.07, +22.14] | [+10.19, +22.68] |
| unsafe | translation+wrist − stream0 | +5.71 [+0.71, +10.71] | [+0.31, +10.80] |
| unsafe | translation+wrist − stream5 | +5.71 [+0.71, +10.71] | [+0.31, +10.80] |

The historical seed-0 opening proposal prefix matched for 253/280 starts. The archive readback joins case keys, initial robot states and exact noise seeds. The historical contrasts compare protocols that also change the support layout, when correction starts and whether the estimate updates during scoring. The fresh wrist-minus-translation contrast isolates the registered wrist intervention.

## Physical and scheduler costs

| Stage | Explicit resets | Verified settling controls | Executed controls | Live VLA batch calls | Replayed opening batches |
| --- | ---: | ---: | ---: | ---: | ---: |
| support | 280 | 2800 | 4200 | 140 | 0 |
| translation | 280 | 2800 | 64497 | 1861 | 0 |
| translation_wrist | 280 | 2800 | 54008 | 1696 | 70 |

Support unsafe events occurred on 0/280 supports. Either support or query was unsafe on 121/280 translation and 91/280 translation+wrist deployments. The seven rollout attempts physically executed 288 support resets, 288 translation resets and 296 wrist resets. They used 4,320 support controls, 65,069 translation controls and 54,008 wrist controls. Relative to the paired panel, this adds 32 resets, 320 settling controls, 120 support controls and 572 translation controls. No unconfirmed interactions remain in the journal.

The last launcher exited 0 at 17:27:07 Shanghai. Baige recorded 1.762 GPU-hours across the seven native scoring submissions and 0.219 GPU-hours across three renderer diagnostics, for **1.981 GPU-hours** in total. [Allocation accounting](evidence/scheduler-cost.json) · [Scoring receipts](evidence/scheduler-scoring-allocations.jpg) · [Diagnostic receipts](evidence/scheduler-render-allocations.jpg) · [Final job](evidence/scheduler-retry07-complete.jpg)

| Attempt | Baige job | Launcher exit | Sealed batches | Launcher GPU-hours |
| --- | --- | ---: | ---: | ---: |
| retry01 | job-iynjhzjijoyj | 2 | 0/70 | 0.0867 |
| retry02 | job-7nq10tvhsewa | 1 | 54/70 | 0.9697 |
| retry03 | job-bg6cc5kozt62 | 1 | 0/70 | 0.0011 |
| retry04 | job-0ttm7yo8nha0 | 1 | 0/70 | 0.2656 |
| retry05 | job-c2yodm5r90kh | 3 | 14/70 | 0.1944 |
| retry06 | job-q90tsmxqn5z6 | 0 | 1/70 | 0.0600 |
| retry07 | job-xlb1k8c5yfgs | 0 | 1/70 | 0.1078 |

The attempts used 1.6853 launcher GPU-hours in total. Retry02's retained support and translation are counted once. The incomplete retry02 wrist attempt and rejected reset candidates are physical cost only. [Original failure](RESULT-NATIVE.md) · [Full audit and case rows](evidence/retry02/full-readback.json.gz) · [PFS progress](evidence/retry07/monitor.jsonl)
