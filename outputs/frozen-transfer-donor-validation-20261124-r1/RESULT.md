# Fixed IIWA trajectory transfer on separate Kinova3 drawer layouts

## Five layouts with archived frozen-policy baselines

The development-selected IIWA tape 35 completed **1/5** layouts on Kinova3. The raw frozen π0.5 tapes completed **2/5**, both live and in matched CPU replay. Tape 35 completed row 31, which the frozen policy already solved, and lost the success on row 16. It rescued **0/3** historical raw failures. The shortest successful IIWA tape, 34, also completed **1/5**. It rescued row 14 and lost both original successes. Each donor supplied exactly the same source controls on all targets. [Registration](PLAN.md) · [Input identities](inputs/manifest.json)

| Released row | Historical raw | Own raw replay | IIWA 35 | IIWA 34 | Historical tuned Perceiver | Own tuned replay |
|---:|---|---|---|---|---|---|
| 14 | H300 | H300 | C184 | **S171** | — | — |
| 15 | H300 | H300 | C184 | C171 | — | — |
| 16 | S206 | S206 | C184 | C171 | — | — |
| 17 | H300 | H300 | C184 | C171 | — | — |
| 31 | S241 | S238 | **S179** | C171 | S180 | C180 |

S gives the one-based control receiving native reward one. H is a native horizon truncation. C is a right-censored tape end with zero reward observed. A dash indicates that the registered archive supplied no such branch. [Historical raw on 14/15](../frozen-vla-native-transfer-20261123-r1/RESULT.md) · [Historical raw on 16/17](../frozen-vla-tracking-feedback-20261123-r1/RESULT.md) · [Row-31 raw and tuned](../frozen-transfer-progress-20261124-r1/NATIVE.md)

Comparing completion within each full tape, donor 35 versus own raw had zero gains, one loss and four ties (−20 percentage points). Donor 34 versus own raw had one gain, two losses and two ties (−20 points). Donor 35 versus 34 had one gain and one loss on different layouts (0 points). These count observed reward within the issued sequences. On four of five layouts each donor ended earlier than any native success or official timeout. Their hypothetical returns with more actions remain unknown.

The time budget changes the comparison. By control 184, donor 35 had completed row 31 and raw had completed none. By 171, donor 34 had completed row 14, while donor 35 and raw had completed none. Thus 35 finished an already solvable layout earlier, and the source-only control produced the only rescue. On row 31 the CPU-tuned tape had no reward by its 180-control end, although the original live tuned branch succeeded at 180. This single local comparison cannot establish an advantage over the tuned policy.

## Eight fresh layouts with fixed tapes only

Tape 35 completed **2/8**, on rows 46 and 47. Tape 34 completed **3/8**, on rows 41, 43 and 46. For full recorded tapes, 35 had one exclusive completion, 34 had two, and the other five rows tied (−12.5 percentage points for 35). At equal exposure through control **171**, the counts were **0/8 versus 3/8**. Both completions under tape 35 occurred after control 171. Raw frozen-policy performance on these eight layouts remains unmeasured. [Paired counts, contrasts and common budgets](evidence/summary.json)

| IIWA donor | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | Observed completions | Censored |
|---|---|---|---|---|---|---|---|---|---:|---:|
| Selected 35 | C184 | C184 | C184 | C184 | C184 | C184 | S184 | S177 | 2/8 | 6/8 |
| Shortest 34 | C171 | S171 | C171 | S171 | C171 | C171 | S168 | C171 | 3/8 | 5/8 |

## Interpretation and reproducibility

The 3/4 development result for tape 35 fell to 1/5 on separate layouts with a raw baseline. The shorter source-selected tape produced the only rescue and completed more of the eight additional layouts. These observations give no basis for promoting the development-selected tape over the frozen policy or the shortest control. They do show that a fixed IIWA trajectory can sometimes earn native drawer reward on a different embodiment. The two tapes reach success at different times, and their source-conditioned controls, release layouts and policy noise differ together. This design cannot attribute their differences to any one of those factors. Both tapes were selected from successful source episodes. All policy weights stayed fixed. [Development matrix](../frozen-transfer-donor-20261124-r1/RESULT.md) · [Selection rule](PLAN.md)

Each layout contributes one paired observation on one task. The original raw outcomes on the five archived rows were known when the test was registered. The donor outcomes were new. As an indication of sample size, descriptive 95% Wilson intervals for observed completion on the eight fresh rows are 7–59% for tape 35 and 14–69% for tape 34. On the five primary rows the paired sign comparison for 35 against raw contains only one discordant row (two-sided exact p = 1 under sign symmetry). Fixed rows from one release do not establish a population success rate. [Counts and uncertainty](evidence/summary.json)

The CPU matched the official release row, seed, BDDL, pre-settle scene and perturbation to each archived target. Across the branches on each of the 13 layouts, the entire local reset, body identity and initial public observations matched on the first attempt. None of the five historical GPU initial physics or post-settle scene hashes equaled the corresponding CPU hash. All five raw success labels transported, with row 31 three controls earlier. The historical tuned success did not transport within its recorded budget. CPU replay uses the original closed-loop policy's already-issued actions and makes no fresh inference from the CPU observations. The development host used Mesa llvmpipe while the original policy sessions ran on separate RTX 4090 workers. Each local shard had single-core affinity and one thread configured for BLAS, OpenMP, MKL, Mesa and NumExpr. The historical launchers likewise set one thread for BLAS, OpenMP, MKL and Mesa. Host and numerical drift remain relevant even under those settings. [Per-case resets and native endpoints](evidence/cases/) · [Original writer and native module hashes](evidence/provenance.json)

The preregistered validation ran at Git revision `311b4f8a24ace9602f8bb5dfc818155489484287` on Baige CPU 30/31, 11:54:21–11:57:45 Shanghai, November 24. Both shards exited zero in **204 seconds** with **6,124 controls** and **zero GPU-hours**. Across 32 branches there were nine successes, three native horizon timeouts and 20 censored tape ends. All seven float32 channels were checked byte for byte at the native simulator call. The portable reader rehashed each full tape and consumed prefix. The physics pin was `a8a997e7d06bb131d3f9710204c58e106f0f0904`, and the release pin was `e5500c690c1498eca1b7050350654964190cdf70`. The CPU study is complete. [Pinned launcher](run.sh) · [Readback](score.py) · [Baige exit receipt](evidence/exit.txt) · [Transferred evidence hashes](evidence/BAIGE-SHA256SUMS)
