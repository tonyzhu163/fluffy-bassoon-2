# First capped source stream

Baige `job-p5we8pk45d8i`, `libero40-1m-sac-20261029-s00-train-r1`,
source `8ada0c0a59ea562716b251dc3cd7d71dc1ba3b18`, exited0.
The unchanged official OpenPI40 motor completed80 episodes/20 cases on
`libero_spatial:00`, with20 actual distinct nominal state vectors/IDs0–19.
Seventy-nine episodes succeeded.742 query chunks and4,920 SAC updates;
7,082 primitive+810 settling=7,892 total physical actions. No unresolved
RPC charge. The9,500-action cap stopped before another worst-case K4 case;
all started cases finished, with1,608 reserved actions unused.512 candidate
cases were not a required collection count.

Both final AD/DPT exports contain the same742 actual history chunks;
publishing prefixes does not multiply the interaction count. Source recovery,
replay and optimizer/RNG state remain on Baige.191 focused Linux tests passed,
two JAX tests skipped in the probe environment; actual typed/legacy JAX RNG
preflight passed. Worker723.91s on NVIDIA RPBZZZ6, scheduler13m10s.

This qualifies one capped source stream. It is neither all40 coverage nor
evidence of teacher improvement, useful recovery, memory or transfer.
The [index](index.json) preserves episode/cost/coverage receipts.
[Independent inspection](independent-artifact-audit.json) verified all80
trace hashes, both paired-prefix datasets/source checkpoints and the final
recovery hash, with consistent action/reward lengths and terminal boundaries.
