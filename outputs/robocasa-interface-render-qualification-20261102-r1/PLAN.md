# Rendering qualification before a fresh Stage 0 allocation

Prepared2026-11-02; no new GPU allocation submitted. The failed Stage0 job
`job-16bb8e4z2n3x` is terminal on all four instances. Keep its source, case
bank, guard and artifacts immutable.

The existing one-level guard rejected TurnOffStove native seed241109. CPU
comparison finds exact physical/model/camera/public state; four right-camera
pixels (12 color components) differ by2, while wrist pixels are identical.
The observation was not modified. This exceeds the qualification; it is not
yet permission to widen the bound or admit that reset.

Proposed next action: oneGPU/32CPU/128GiB/16GiBSHM, ten scheduler minutes,
at most720GPU-seconds charged including startup/closure. This comes from the
original12GPU-hour allowance: the failed run is conservatively charged2924
of43200GPU-seconds, leaving40276 before qualification. No PPO or VLA inference.
No automatic continuation into the scientific screen.

Reconstruct all80 selected TRAIN failure cases from the frozen failure banks,
four independent fresh constructor/first-reset repeats each (320 resets),
using the same native source, assets, cameras, seeds, rendering and original
unmodified pixels. Check exact physical/compiled-model/camera/public state
against each original canonical snapshot. Record pixel delta distributions
without changing the production guard. Charge reset controls, all probes,
startup and failed attempts. Allocation-only CUDA probing and rendering stay
on the scheduler GPU. Stop at the finite deadline and retain every snapshot.

After collection, independently summarize on allocated Linux CPU. A revised
numeric bound requires an explicit saved qualification and a fresh scientific
run contract; never retrofit the failed run. A negative physical/state match
requires a simulator diagnosis, not a larger pixel allowance.

Before any scientific continuation, qualify the optional complete-episode
checkpoint hook and use it to save actual chunk records at every episode,
including successful siblings when another lane fails. Persist motor/control
plans before dispatch. Retain existing complete valid native outcomes without
resampling by success, disclose their missing chunk records, and never use
them as AD/DPT histories. A new output identity must enumerate exactly the
missing episodes, partial-call charge and source migration.

The approved Stage0 contract says **"No automatic fresh allocation."** This
qualification is prepared for a fresh-allocation decision; it has not launched.
