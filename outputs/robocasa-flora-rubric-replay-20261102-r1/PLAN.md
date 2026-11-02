# Frozen rubric replay on retained videos

The human requested a comparison of the new generalized visual completion
rubric against the original prompt, especially failed episodes originally
given full-success scores. Freeze the already written candidate without
editing it in response to replay results.

Replay all384 completed Flora-PPO TRAIN episodes and the earlier five50-video
calibration banks (634 videos total). Run both original and candidate prompts
on every video (1268 labels), prioritizing the58 TRAIN failures whose saved
greedy score was5. Retain every case, including true successes and ambiguity.
These are development videos, not an independent qualification set.

Keep the official code/model revisions, original video bytes/camera order,
native instruction, terminal frame sampler including the final frame,
temperature0/seed0, structured output, five grade probabilities and
`(expected_score-1)/4` reward calculation. Append only the frozen shared and
current-task criteria to the task text passed through the official terminal
message helper. The original native instruction remains an exact prefix.
Original and candidate requests for each video share the same persistent
engine and inference batch. Scorer jobs contain no simulator truth, old
grades, native poses, distances, joint values or audit notes. Truth and saved
grades are joined only by the CPU summary.

Bind every source receipt, all video hashes and both rendered prompt digests
to a new run identity. Never reuse production label caches. Compare saved
original versus rerun-original separately, so numerical replay changes are
visible. Existing policy runs and prompt contracts remain immutable.

Report per-cohort AUROC, Score5 false-positive rate/precision/recall, all
58 original full-score failures and their paired score/probability/reward
changes, new false positives, lost true full scores and low-scored successes.
Report exact paired correction/regression counts and tests. A changed reward
map, new PPO training, new episodes or final held-out policy evaluation are
outside this replay.

Bounded Baige allocation:2 nodes x2 GPUs,32CPU/256GiB RAM/16GiB SHM per node,
55 scheduler minutes,3300 inclusive worker seconds per node,13200 GPU-seconds
maximum,16GiB output. Four TP1 engines, batch8, memory utilization.78. No
automatic retry; retain partial rows and charges on failure. Current observed
experiment commitments are32 GPUs; this request brings them to36, within the
existing40-GPU aggregate authority (development container excluded).

Use current reviewed main, Git-deployed immutable source, allocated GPU probes,
CPU numerical/core checks and actual progressing replay rows before claiming
launch. No new simulator episode is needed for a scorer-only video replay.
