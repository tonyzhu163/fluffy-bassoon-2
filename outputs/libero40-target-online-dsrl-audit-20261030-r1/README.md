# Fresh target-online DSRL learning onset

Baige `job-hweficar0ilk`, run `libero40-target-online-dsrl-20261030-r2`,
source `0c897640033373b1f3efbac63830ff6a626b0292`, exited0.59 allocated Linux
checks and the actual GPU probe passed. Original preflight-r1
`job-o2odrn14qpsm` stopped on a test error-message assertion before model
initialization, reservation or robot actions;49 seconds retained. The corrected
retry had a43-minute scheduler ceiling, within the45-minute initial slice.

Fresh independent pixel-SAC actor/critic/optimizer/replay on each original
LIBERO90 task0/1,state0,scene seed123; official frozen OpenPI40 motor. Published
batch256/UTD20, first update only after replay>500 chunks, fixed actor during
each episode. Collection warmup is tied32D Gaussian. Published evaluation is
independent H10 Gaussian before updates, stochastic SAC afterward.

| Case | Train episodes / successes | Train chunks | SAC updates | Actual training actions | All actions |
| --- | ---: | ---: | ---: | ---: | ---: |
|Task0 |15 /1|583|2,400|5,976|7,626|
|Task1 |15 /0|600|2,400|6,150|7,800|

All eight separate frozen queries failed: both cases after0/1/13/15 training
episodes. The13-episode queries had800 optimizer updates each;15 had2,400.
Actual support-action amounts vary with early success. Queries never entered
replay or changed subsequent training randomness. This is learning onset,
not a converged DSRL result or proof that ordinary DSRL cannot solve the tasks.

Total15,426 executed actions within16k includes30 training episodes, eight
query episodes and20 initial settling/qualification actions; no unresolved
charge. [Independent audit](independent-artifact-audit.json) verifies all38
episode trace hashes, costs/boundaries, all eight immutable query-recovery
hashes and actor/optimizer signatures, both final replay/RNG recoveries, and
exact initial public-array matching across query milestones. Raw images,
RPC payloads and complete recoveries remain on Baige. [Root result](index.json).

**Numerical limitation:** query0 has exactly the earlier native control's
initial RGB/wrist/proprio arrays, stochastic seed and all native noise arrays,
but first motor actions differ by max0.004149556 (task0) /0.002440870 (task1)
on the same physical GPU. Thus full trajectories are not bitwise paired
across these runs. A bounded zero-action motor replay is checking typed versus
legacy JAX keys and runtime initialization order. Do not attribute this cause
without evidence, or infer a tightly paired meta-versus-DSRL advantage yet.
