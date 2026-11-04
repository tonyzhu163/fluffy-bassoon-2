# Partial checkpoint curve, 2026-11-04 13:22 Shanghai

This is a provisional DEV readout, not the selected-policy result. Only
completed eight-bank panels are included: 64 fixed fresh cases per task and
checkpoint, native first success within 600 controls. The evaluation traverses
130 down to zero; the plot reads left to right in training order. At this
snapshot the five jobs were evaluating checkpoint80, with no winner locked.

| Task | 90 | 100 | 110 | 120 | 130 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenDrawer | 48 | 48 | 48 | 46 | 47 |
| CloseToasterOvenDoor | 23 | 26 | 23 | 19 | 25 |
| PickPlaceDrawerToCounter | 33 | 28 | 24 | 32 | 21 |
| CoffeeSetupMug | 17 | 23 | 19 | 25 | 26 |
| TurnOffStove | 10 | 4 | 7 | 7 | 7 |

All counts are out of64. [Raw snapshot with bank SHA256s](CURVE-PARTIAL-20261104-132251.json)
and [plot](CURVE-PARTIAL-20261104-132251.png). The shaded pointwise95% Wilson
intervals describe case-sampling uncertainty; they do not include all repeated
trajectory variability or correct for checkpoint selection. Lines are visual
guides, not fitted curves. Rendered with Matplotlib3.10.7 on a separate Linux
CPU plotting environment; no model computation or active-runtime changes.

Drawer-to-counter has the largest earlier-versus-final gap in this window:
33/64 at90 versus21/64 at130,18.75 percentage points. Its rebound at120 makes
this an uneven trajectory, not a monotonic collapse. OpenDrawer is nearly
flat; Coffee improves over this window; Toaster and Stove are uneven.
Checkpoints0–80 were pending at the snapshot, so it cannot establish when
learning converged or whether any checkpoint improves on the initial policy.

## Hypotheses and stopping

The source used native simulator first-success reward, not Flora reward.
Learned-reward mismatch is therefore not an explanation for these runs.
One hypothesis is continued noisy actor updates shifting useful pretrained
behavior: actor LR stayed5e-6, with four PPO epochs and no KL penalty or KL
early stopping. PPO clipping does not guarantee monotonic robot performance.
Another hypothesis is the difference between stochastic Flow-SDE training
(additional noise at one denoising step) and ten-step Flow-ODE evaluation
(without that additional noise). ODE refers to action generation, not robot
simulation; initial sampled noise is still present and its RNG is shared
across checkpoints. Neither mechanism is established by this curve.

[Both actor ranks' existing JSON metrics](DRAWER-OPTIMIZATION-PRELIMINARY.json)
show no simple exploding-loss/critic-collapse signature: updates81–90 versus
121–130 have native approximate KL0.0796→0.1019, clip fraction0.2127→0.2309,
value loss0.0630→0.0618 and explained variance0.2711→0.2797. These are equal
rank/update descriptive averages; native signed chunk-conditional approximate
KL is not full-action KL. Pre-clipping gradient norm is not evidence of an
unclipped explosion. This audit does not diagnose the causal source of the
performance gap or prove an implementation correct.

For future policy optimization, validation-based checkpoint retention and
predeclared stopping patience are sensible. Validation stopping across
training updates is distinct from KL stopping within one PPO update.
This campaign's130 updates also fixed a corpus collection budget; preserve
that complete trajectory and its original failed teacher gates. Finish all
DEV checkpoints, lock the winner including policy0 as a candidate, then use
independent counterbalanced TEST128 to assess selected-policy performance.
No training, hyperparameter changes or extra evaluation arms were launched.
