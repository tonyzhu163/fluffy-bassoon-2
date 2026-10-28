# CTM T1 throughput integration and independent audit

Requested scope: fix the CTM T1 training bottleneck before expanding the RL
benchmark runs. Reuse the completed, immutable paired timing qualification;
do not launch a duplicate GPU job or change its active source.

Reviewed main `65b48a457f9cf1c9c9dcd1c9813ac9e53d5e0455`, then merged the
qualified implementation at `91c16961a02aa739c6cda6cb3c8627145d51aeef` into
this detached worktree. Integration commit is
`db507380b6961825bfb00f12b2585594fa9919fe`.

Audit saved Baige artifacts, the exact source, the serial-reference tests,
recurrent gradient boundaries, achieved optimizer work and admission receipt.
Only static inspection and artifact arithmetic run on macOS. All model tests
and timing were performed in the existing allocated Linux job.

Keep full-trial gradients as the default. Expose cached 100-step replay only
as an explicit TBPTT protocol, retaining forward memory across the trial.
Learning and success versus GPU-hour comparisons remain a subsequent gate.
