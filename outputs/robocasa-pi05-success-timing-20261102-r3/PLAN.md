# Native timing: pack the unchanged eight workers onto two GPUs

R2 job `job-hqimwtw7znm3` remained queued without allocation, actions or GPU
charge. Cancel it after verifying ownership and zero activity. R3 uses one
two-GPU Baige node instead of waiting for four nodes. Run the same eight
logical workers in four sequential waves; collector bytes, per-worker cases,
seeds, batch sizes, full horizons and p99 calculation remain identical to R2.

No further software test suite: user requested less testing. First-success
logging already passed the retained Linux qualification; only scheduler packing
and clock change. On-allocation checks and actual episode monitoring remain.

One2GPU/64CPU/256GiB node,60minutes, at most7,200GPU-s; with closed R1 cost726,
cumulative maximum7,926 stays below the original9,600. Same250 episodes,
180,000policy controls,212,000all-in ceiling and64GiB outputs. No optimizer,
no automatic retry, no modification of an allocated run.
