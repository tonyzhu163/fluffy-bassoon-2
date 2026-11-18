# Learning histories exploration

Run the protocol introduced in `20622b2722641aa2dc0b051944c7d3dc43dd1410`.
This is the synthetic hidden-rule exploration, not a robot experiment.

- Conditions: demo, blind, attempts, corrections, linked; seeds 0, 1, 2.
- K=6, 20,000 updates, batch 256, AdamW / OneCycle peak learning rate 0.0003.
- Four-layer, width-128 Transformer; unchanged posterior-sampled training attempts.
- Evaluate 8,192 sessions per training/held-out split using the frozen learner's
  own greedy attempts. Keep learner, Bayes-on-learner-history and oracle curves.
- Execution-only additions: atomic resumable checkpoints every 500 updates,
  and evaluation chunks of 256 with the same sessions and labels.
- Qualify checkpoint continuation and evaluation chunk equivalence on allocated
  Linux CUDA, then time a 500-step fit before launching the full grid.
- Revised after qualification: 15 one-GPU workers, one condition/seed per worker,
  for at most 90 minutes each, plus two qualification allocations capped at
  30 minutes each: 23.5 GPU-hours maximum reserved across all attempts.
  Count all attempts, including failures, before any retry.
- Record exact deployed commit, GPU probe, resource checks, fit logs, checkpoints,
  and clock receipts under this run. Never overwrite an output identity.
- Read the README's decision criteria on all seeds; treat blind-versus-demo gain
  as evidence of information in actor choices, not outcome learning.

Initial resource check (Nov 11, approximately 12:55 Shanghai): Baige 4090 queue
displayed 4 free / 152 maximum GPUs with 10 queued jobs. The mounted project PFS
had about 102 GB available. The development container exposes neither `quota`
nor `lfs`; quota is unavailable and capacity/write checks are recorded at launch.
These are observations, not future availability guarantees.

At 13:04 Shanghai, qualification `job-048pexf3qby6` passed on an RTX 4090:
checkpoint continuation was bitwise identical; chunked evaluation preserved all
choices with maximum logit difference 6.86e-7. Batch 256 completed 500 updates in
45 seconds (approximately 0.09 seconds/update). The queue showed 22 free GPUs
and no queued jobs, so the grid will use 15 independent workers to reduce wall
time without changing any fit. Estimated training is 30 minutes per fit, plus
startup/evaluation. The grid launcher additionally waits for every worker before
the master exits; model, qualification and probe code are unchanged.

The first qualification (`job-7l8kllhsi9k0`) failed before training because the
launcher omitted PYTHONPATH for the GPU probe. Its process receipt records 3
seconds; scheduler displayed 49 seconds before cleanup. It was stopped to
release the abnormal allocation. Corrected qualification used 159 process
seconds. Use scheduler durations for conservative allocation accounting.
