# Residual-noise probe: static feasibility only, on hold

**On hold.** Prioritize ready source collection, existing training and the already
prepared readout. No residual profile, launcher or tests were completed; no model,
simulator, allocation, reservation or GPU spending occurred. This is not a gate
for the source-SAC campaign or its approval.

At reviewed source `6dcc0762b70a53b9aef85b1eb5cf357c0e250e11`:

- `vla_libero.FrozenOpenPI.infer` already passes an explicit full H×32 noise array
  to the qualified OpenPI policy; no new motor framework is needed for H10×32.
- `vla_source40_evaluate.rollout_episode` already uses explicit
  `default_rng(episode_seed).standard_normal((10,32)).astype(float32)` for its
  baseline. Label any future comparator **explicit full Gaussian/custom-noise**;
  do not claim bitwise equivalence to `noise=None` or the earlier native31 score.
- The learned-controller branch enforces32D, clips to[-1,1] and ties that vector
  across H10. Directly routing a residual through it would silently test
  `Z+clip(mu)`, not the proposed unbounded `Z+mu`.
- `HistoryNoisePolicy` has an unbounded Gaussian mean head. A future residual
  probe therefore needs a narrow explicit applied-flow-noise protocol, exact
  full Z/mean/applied-noise and RNG receipts, and a declared32D feedback
  convention. One plausible convention is feedback of actual residual alpha*mu,
  with full H×32 motor noise traced separately; that is a changed deployment
  parameterization, not a trained residual method or evidence of memory benefit.
- The existing common-runtime runner and fixed development tasks0,10,...,80 can
  support a future36-query four-condition comparison without outcome selection.
  No runtime/throughput qualification was performed for the proposed1180-second
  or16000-action ceilings, and no such diagnostic budget is reserved.

A tiny uncommitted hook had begun before the hold message arrived. Those edits
were removed from the source tree and preserved only as **untracked local WIP**
in this isolated worktree's output directory. No implementation is delivered or
admitted for launch by this handoff.
