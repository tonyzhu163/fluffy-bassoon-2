# Benchmark-specific CTM warm starts

This is a **DPT-inspired chronological recurrent adaptation**, not an exact
reproduction of DPT, Vintix II or VLRLA. Direct full-trial PPO remains the
teacher-free baseline. No previously trained recurrent teacher is required.

## Data and objective

The public history contains observations, **executed** actions, rewards and
attempt boundaries. A separate training-only labeler supplies the competent
action at each query observation. Labels never replace the actions that
generated the history. Memory and gradients span all attempts in a trial;
each new trial starts from fresh recurrent state.

- Compact tasks: generate contexts with a per-trial mixture of uniform random
  actions and a public-feedback heuristic. The random fraction is fixed within
  the trial, avoiding a prescribed improvement schedule tied to the clock.
  Labels use the known training task: best bandit arm, shortest grid path,
  signed-gain reaching controller, or train-scene population success rates
  given the cached rotation. Cross-entropy / squashed-Gaussian NLL trains the
  existing actor. `--objective behavior_cloning` is an explicit control that
  predicts the executed behavior action instead.
- Native pi0.5: collect imperfect trials with the frozen motor plus fixed
  Gaussian residual noise. At each visited observation, query the frozen
  motor with the same native initial noise and undo the training actuator's
  known XY rotation **only in the action label**. Verify the native decoder's
  affine XY map and unchanged remaining action dimensions. Record an
  independent flow-matching sample `x_t=(1-t)*a_star+t*epsilon`, with target
  velocity `epsilon-a_star`. The learner reads native KV and public feedback;
  its velocity residual is supervised on the same seven active model channels
  and five executed action steps subsequently used by PPO. The native model
  emits eight steps, while its decoder exposes five: relabel only that decoded
  prefix, retain the model-action tail unchanged, and exclude the tail from the
  loss. Frozen motor weights are never optimized.

Native label-policy evaluation is privileged and must be called an oracle
reference, not a learned ICRL result. Measure its competence against the paired
uncorrected motor before spending on supervised training. The labeler is our
controlled-environment extension; the papers do not validate this exact bridge.
The current offline implementation caches one flow-noise/time sample per
history position, so its data manifest also fixes those samples.

## Checkpoints and transition to PPO

Each output directory contains immutable datasets or a checksum-pinned history
manifest, metrics, raw evaluation arrays where applicable, resumable `latest.pt`,
and a portable `warmstart.pt` with a JSON sidecar. The portable artifact contains
weights, source commit, benchmark/interface identity, architecture, data hashes,
cost counters and evaluation. It does not contain transient trial memory.

`--warmstart PATH --warmstart-sha256 SHA` loads these weights in `meta_rl.train`,
`meta_rl.maniskill_rotation`, or `meta_rl.pi05_bridge_train`. The loader rejects
different families, observation/action contracts, architectures, cached outcome
tables, native motors, normalization files and relevant RLinf source files.
It starts a fresh PPO optimizer and RNG stream and resets the critic to zero.
`--resume` instead restores the complete same-phase optimization state; repeat
the initialization arguments when reconstructing that run's contract.

Portable artifacts are candidates, not certificates of adaptation. Report
normal, memory-wiped and reward-blind frozen-weight behavior. Compact evaluation
includes greedy and sampled actions. Current compact seed splits separate
interaction realizations, not all latent grid goals or signed-gain values;
do not call them held-out-task generalization. Native supervised validation
holds out complete history batches from the training distribution. Live
held-out-task evaluation remains a separate required check.

## Entry points

- `python -m fluffy_bassoon.meta_rl.warmstart_train --task TASK ...` generates
  compact DPT data and trains one initialization. Use `--dataset-dir` to reuse
  the exact saved data when resuming or comparing architectures.
- `python -m fluffy_bassoon.meta_rl.pi05_bridge_train --collect-dpt ...`
  records native histories and paired base/oracle competence evaluation;
  it performs no PPO updates. Use a fresh output identity; dataset collection
  is not resumable and its fixed label-noise seed is recorded.
- `python -m fluffy_bassoon.meta_rl.pi05_warmstart_train --histories MANIFEST
  --histories-sha256 SHA ...` starts a fresh native bridge on those histories.

All model computation and tests require allocated Linux execution. Dated run
sizes, resource caps and measured results belong in `outputs/<run-id>/`.

## Sources and differences

[DPT, Algorithm 1 and Section 3](https://arxiv.org/html/2306.14892) separates
context generation and task-specific optimal action labels.
[Vintix II, Sections 3.1–3.3](https://arxiv.org/html/2604.05112v1) extends this
with imperfect contexts, demonstrator labels and a flow-based output
distribution. Both use query-conditioned transformer contexts; our CTM
consumes chronological streams instead.
[VLRLA, Section 4 and Appendix A](https://arxiv.org/html/2610.19517v1) trains
a recurrent bridge from learning histories, zero-initializes its residual
output, and retains its frozen motor. Its KV-residual architecture differs
from our velocity residual.
