# OpenDrawer completed-state wording: measured negative result

The retry completed on Baige `job-of3hcwyigjtu`, source `cbcffb71`, exit0.
Only14 retained OpenDrawer videos were scored:10 failures and4 success
controls,28 labels comparing revision1.1 with the human-selected visual
completion rule in revision1.3. No other task or full-bank replay ran.

| Graded rubric | False successes out of10 failures | Success controls accepted out of4 | Previous false successes corrected | New false successes |
| --- | ---: | ---: | ---: | ---: |
| Previous revision1.1 |7 |3 |0 |0 |
| Visible completed state, revision1.3 |8 |3 |0 |1 |

The new rule does not improve these selected examples. Every prior false
success remains a false success. One formerly Score4 failure becomes Score5:
`flora-single-case-000272-rank-02-episode-000004`, expected grade4.3060→4.3720.
The other13 hard grades are unchanged. Neither success-control losses nor
recoveries occur. The previous1.1 control reproduces all14 retained hard grades,
so the result is not a changed control label set. Exact outputs and expected
grades are in [PAIRED-SAMPLE.json](PAIRED-SAMPLE.json).

The tested clause is specifically for **opening**: the upper target drawer
must visibly appear fully open at the observed endpoint before Score5 is
awarded; gaps, exposed contents and obvious partial extension are insufficient,
while1–4 remain partial-progress grades. The shared rubric is unchanged.
The saved CloseDrawer closing example is inactive and was not sent to these
OpenDrawer videos. [Exact candidate](revision1_3.json).

Do not adopt1.3 as a reward repair or infer a population false-positive rate
from ten error-enriched failures. This is a small development diagnosis and
does not qualify first-success timing,600-control calibration or PPO reward.
The persistent errors still need an explanation tied to the native target and
slide-joint condition. This test does not reveal whether individual failures
come from target selection, insufficient displacement or a later undone state.
The old bank does not retain final drawer-joint values or native first-success
timestamps. No additional model-scoring run was launched after this result.

## Per-trajectory comparison

Native IDs below omit the common TRAIN/OpenDrawer prefix. Values are literal
one-to-five grades and official expected grades, not success probabilities.

| Native episode ID | Simulator | R1.1 grade | R1.3 grade | R1.1 expected grade | R1.3 expected grade |
| --- | --- | ---: | ---: | ---: | ---: |
| `flora-single-case-000112-rank-06-episode-000001` | failure | 5 | 5 | 4.8124 | 4.8583 |
| `flora-single-case-000272-rank-02-episode-000004` | failure | 4 | 5 | 4.306 | 4.372 |
| `flora-single-case-000064-rank-00-episode-000001` | failure | 5 | 5 | 4.8972 | 4.9376 |
| `flora-single-case-000376-rank-07-episode-000005` | failure | 5 | 5 | 4.742 | 4.7346 |
| `flora-single-case-000256-rank-00-episode-000002` | failure | 5 | 5 | 4.0511 | 3.7624 |
| `flora-single-case-000216-rank-03-episode-000001` | failure | 5 | 5 | 4.1834 | 4.3345 |
| `flora-single-case-000208-rank-02-episode-000001` | failure | 1 | 1 | 2.9448 | 2.7493 |
| `flora-single-case-000088-rank-03-episode-000005` | failure | 5 | 5 | 3.2281 | 3.2698 |
| `flora-single-case-000336-rank-02-episode-000000` | failure | 5 | 5 | 3.9568 | 3.9473 |
| `flora-single-case-000024-rank-03-episode-000005` | failure | 2 | 2 | 2.5667 | 2.6503 |
| `flora-single-case-000128-rank-00-episode-000006` | success | 5 | 5 | 3.6278 | 3.2733 |
| `flora-single-case-000072-rank-01-episode-000000` | success | 1 | 1 | 2.2189 | 2.3823 |
| `flora-single-case-000264-rank-01-episode-000001` | success | 5 | 5 | 4.3334 | 4.1723 |
| `flora-single-case-000144-rank-02-episode-000004` | success | 5 | 5 | 4.8147 | 4.8655 |

## Validation and cost

The model saw native instruction, frozen criteria and the same three raw RGB
videos only; truth and cached grades were read only in the CPU summary.
All42 video hashes and all four batch hashes match their receipts; all28
labels use the unchanged official64-frame sampler and1–5 system/parser.
One TP1 engine was loaded once, batch8, memory utilization.78, temperature0.
The terminal worker is byte-identical to the earlier eight-test qualification.
The retry's source migration745f10f1→cbcffb71 changes no launch/scorer/rubric
or terminal-worker bytes. [CPU checks](CPU-CHECKS.json),
[retry checks](RETRY-CPU-CHECKS.json), [raw result](RESULT.json).

Worker charge is531 GPU-seconds. The scheduler reports9min10s running,
22:55:29–23:04:39 Shanghai. Even charging submission-to-terminal gives563
GPU-seconds, below the900 cap. The job succeeded and released its one GPU;
there were no robot actions, simulator episodes or optimizer steps. Earlier
UI service failures remain separate evidence, not model inference failures.
[Terminal charge](terminal-charge.json), [scheduler view](completed-job.png).
