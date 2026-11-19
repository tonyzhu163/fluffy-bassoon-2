# First LIBERO decision-pretraining pilot

Keep frozen OpenPI π0.5 and extend the retry decision menu. Test the existing
five-control default and ten controls with fourteen waits against two variants.

- **Short waits.** Execute ten controls and wait seven. Under S20, seven
  controls is the configured one-way delay. This trades observation freshness
  for more movement within the native horizon.
- **Gripper-dependent waits.** Execute ten controls and use seven waits
  routinely. Wait fourteen after the first burst or a commanded open/close
  transition. This gives grasp and release commands time to propagate through
  both paths. Use the sign of the public gripper command. Zero is a hold.
  Pass the last commanded gripper value through all waits.

Both new choices run under BASE and S20 without a condition label. Every wait
counts against the released horizon. Rerun both old choices in the same runtime
since startup and simulator IPC changed after paired01.

First measure all four choices on the 32 existing development cases: eight
tasks, two support/query reset pairs and two timings. Start every support
with default execution. Reuse the exact case and prompt construction from
paired01. Each query branch has the same reset, raw tensors, horizon and
paired policy noise. Execute 32 supports and 128 queries.

Collect the same design on the registered eight-task fitting subset with
a separate seed stream. Retain the support's public trace, the frozen
query embeddings and all four native outcomes. Use successful modes
at the saved query as the categorical target, uniform over ties. Failed
queries stay in the result denominator. The timing reader for this pilot
uses the first 80 physical controls of each support to limit length and
terminal-outcome shortcuts. Keep all support tapes for auditing.

Fit a two-layer causal reader (width 128, four heads) across independent
initializations. Train with retained and empty support. Fit a separate
current-observation-only control with the same seed, architecture and updates.
Use no development information in training, including normalization or
checkpoint selection. Score all models on the development branch table.
This is replay of measured native branches. Report retained, cleared,
swapped-timing support, current-only, a fitting-selected fixed mode and
lag selector, and the per-query branch ceiling. Describe paired gains and
losses across tasks and fitting seeds. This fit uses a single paired outcome
per execution mode and is exploratory. The released task IDs stay intact.

One Baige GPU shared under the project device lock for collection. Reserve up
to 2.75 GPU-hours. Fit the small reader on allocated CPU. Keep the official
test split reserved. Use the measured panel to decide whether a fresh live
comparison with a shared query prefix is informative before scaling.
