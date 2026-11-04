# Partial native checkpoint curves, 2026-11-04 14:14 Shanghai

[Figure](CURVE-PARTIAL-20261104-141441.png), [all56 completed points](CURVE-PARTIAL-20261104-141441.csv),
[raw snapshot with every point's eight bank SHA256s](CURVE-PARTIAL-20261104-141441.json).
Each point is native first success within600 controls on the same64 fresh
DEV cases/task, with fixed eight-case banks and common sampler RNG. The plot
uses the full training-update axis; grey regions have not been evaluated at
this snapshot. Evaluation itself continues in the approved130→0 order.

| Task | Best completed checkpoint so far | DEV successes | Update130 | Gap, best minus130 |
| --- | ---: | ---: | ---: | ---: |
| OpenDrawer | 30 (ties70,90,100,110) | 48/64,75.0% | 47/64,73.44% | +1.56pp |
| CloseToasterOvenDoor | 30 | 27/64,42.19% | 25/64,39.06% | +3.13pp |
| PickPlaceDrawerToCounter | 80 | 34/64,53.13% | 21/64,32.81% | +20.31pp |
| CoffeeSetupMug | 60 (ties80,130) | 26/64,40.63% | 26/64,40.63% | 0pp |
| TurnOffStove | 30 | 12/64,18.75% | 7/64,10.94% | +7.81pp |

These are provisional observed maxima, not locked selections or independent
improvement estimates. Earliest ties use the declared selection rule, but
policy0 and the other earliest checkpoints remain pending. No TEST panel had
started. The56 completed panels comprise11/12/11/11/11 checkpoints across the
five tasks; Toaster has20–130, the others30–130. Raw bank progress includes
additional incomplete checkpoint panels:459 banks/3672 episodes in the
14:14 monitor. The curve snapshot contains only complete64-case panels.

OpenDrawer mostly plateaus, with a dip around50 and subsequent recovery.
Toaster fluctuates without a clear sustained rise. Drawer-to-counter has
several earlier checkpoints outperforming130 (40:33,80:34,90:33,120:32,
versus130:21); the rebound at120 rules out a simple monotonic-collapse story.
Coffee's final checkpoint ties the best observed value. Stove remains low,
with earlier30 scoring higher than130. These task differences do not support
one universal late-training collapse.

The main practical evidence is checkpoint sensitivity, strongest for
Drawer-to-counter. It supports retaining validation checkpoints rather than
assuming130 is best. It does not establish convergence, improvement over the
initial policy, or a causal optimizer defect. Selecting a maximum among many
DEV observations biases its apparent advantage; independent TEST128 follows
only after all16 DEV checkpoints and the model-bound selection lock.
Pointwise95% Wilson shading describes case uncertainty, not a paired
checkpoint-difference interval, multiple-selection correction or all
trajectory variability. The existing [optimization hypotheses and both-rank
metric audit](PARTIAL-CURVE.md#hypotheses-and-stopping) remain provisional.

The latest [monitor](MONITOR-20261104-141415.json) found allfive jobs advancing,
with passing probes, zero initial likelihood gaps, recent raw episode/initial/
sampler hashes and common RNG checks, unchanged clean code pins, no errors or
exit clocks,8.265GPUh conservative cumulative creation-to-check upper out of
the22GPUh cap,21.680GPUh full initial reservations and618.32GiB free PFS.
Original final-policy results and failed teacher gates remain unchanged.
