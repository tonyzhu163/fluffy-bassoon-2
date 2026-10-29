# Controller and recovery integration

Baige source `7c7a54fa907e2d1479c8be99643497b0bba97055`, run
`libero40-pi05-history-integration-20261029-r3`, exited0. No new simulator
actions were collected. The original source corpus/recoveries remain intact.

- Focused Linux suite153passed/2skipped. The JAX skips are in the separate
  probe environment; actual JAX source restore/update checks ran on GPU.
- Saved model/optimizer state, sampler and SAC/motor RNG restore exactly.
  Independent GPU next updates are numerically close, not bitwise identical:
  largest recorded array difference3.65e-6, within declared rtol1e-4/atol1e-5.
  Original recovery SHA remains unchanged. [Numerical receipt](recovery.json).
- All four AD/DPT ×full/static controllers completed25 optimizer updates
  from the same source history and disjoint validation states, exporting
  fixed-update `final.pt` and complete `latest.pt` recoveries.
  [Hashes and counts](students.json).
- R1's opaque serialized-byte equality check and R2's tighter numerical
  threshold stopped before student training. Their outputs are preserved;
  exact restoration passed R2, while next-update moments differed by2.82e-6.
  R3 explicitly distinguishes exact restoration from numerical continuation.

These25-update models are technical candidates, not robot-performance or
adaptation results. They do not establish useful source learning, memory
benefit, sample efficiency or corpus sufficiency.
