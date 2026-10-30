# All40 TRAIN corpus: recorded feedback content

Read-only metadata analysis of every canonical TRAIN stream. No goal or case
was excluded; no raw trajectory array, model, simulator or test was executed.
The index paths/SHA256 values and canonical reservation identities are retained
in [summary.json](summary.json). [per-goal.csv](per-goal.csv) gives the40-goal
breakdown. The reproducible stdlib-only script is [analyze_metadata.py](analyze_metadata.py).

There are527 complete K4 cases and2,108 episodes, covering527 goal-qualified
initial-state IDs and527 distinct nominal vectors summed across goals. Goals
contain5–22 actual cases each (20–88 episodes), rather than the advertised
512-candidate case banks. Every four-episode case has a consistent recorded
nominal-state hash. All2,108 receipts lack measured post-settle simulator-state
hashes, so actual post-settle physical-state counts are **unknown**. Nominal
vector hashes, seeds and camera hashes are not substituted for that evidence.

| Recorded behavior | Episodes | Successes | Failures | Steering chunks |
| --- | ---: | ---: | ---: | ---: |
| Tied-Gaussian warmup |1,332|1,256|76|20,436|
| Learned stochastic SAC |776|758|18|10,915|
| Mixed/missing episode stage |0|0|0|0|
| Total |2,108|2,014|94|31,351|

Warmup supplies65.2% of steering chunks and76/94 failures. Learned behavior
has only18 recorded failures across10 goals;30 goals have no learned-behavior
failure in this corpus.12 goals have no failures at all. Warmup and learned
success rates (94.3% and97.7%) are descriptive, not a controlled learning
improvement: they occur on different chronological cases, and learned action
sampling changes from unbounded Gaussian to the learned bounded distribution.

| Actual K4 sequence content | Cases |
| --- | ---: |
| All four episodes succeed (`SSSS`) |454/527 (86.1%)|
| All four fail (`FFFF`) |1/527|
| Any failure followed by a later success |60/527|
| First episode fails, some later episode succeeds |27/527|
| Failure→later success, all four episodes warmup |44|
| Failure→later success, all four episodes learned |8|
| Failure→later success, warmup/learned transition case |8|
| Contains learned failure followed by learned success |10|

The last row includes two transition cases, so it is not an additional
disjoint category. Across the60 cases there are132 ordered failure/success
episode pairs. All60 case IDs, actual patterns, episode stages and optimizer
clocks are retained in summary.json. These are available recorded recovery
sequences, not evidence that memory caused recovery: source SAC weights and
exploration can change between episodes.14 goals contain no such case.

| Suite | Cases | Episodes | Failures: warmup / learned | Cases with failure→later success |
| --- | ---: | ---: | ---: | ---: |
| Spatial |163|652|9 /5|9|
| Object |127|508|14 /1|13|
| Goal |162|648|21 /2|15|
| LIBERO10 |75|300|32 /10|23|

Every receipt includes a stage counter, outcome, nominal state hash and
optimizer-clock value; all stage counters sum to their episode chunk counts.
Recorded `source_updates` is the **post-episode** optimizer clock. The
pre-episode clock is the previous complete episode's value (fresh learner0).
All warmup episodes start at clock0; the threshold-crossing warmup episode
can end at up to800 updates. Learned episodes start at180–6,640 updates.
Final per-task clocks range4,920–7,120; their sum233,580 counts compute across
40 independent learners, not one shared teacher. Update counts establish
optimization activity, not expert quality.

Committed episodes cost304,797 primitives plus21,080 settling actions,
325,877 total. The326,277 training-role ledger charge adds400 qualification
actions. Validation data and costs are absent from these training statistics.

The fixed eight00/01 source-validation goals contributed161 learned-behavior
training episodes with zero failures (six failures occur during their warmup).
That helps explain why this predeclared small source panel offers little
recovery headroom; no new cases were selected using these outcomes.

The measured limitation is the corpus's sparse failure/recovery content,
especially under learned behavior. This is a plausible limitation for learning
feedback-dependent steering, not a causal explanation of the LIBERO90 student
deficit and not a negative result for ICRL. Frozen teacher competence, matched
native controls and actual target queries remain separate robot measurements.
