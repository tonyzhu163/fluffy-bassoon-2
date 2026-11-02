#600-control cutoff decision

The human selected **600 physical policy controls (12 H50 chunks)** on
2026-11-02. This is a provisional operating cutoff anchored by a partial native
pi0.5 timing panel, not a completed population-p99 qualification. The latest
observed first success was538, giving62 controls of margin. It is20% shorter
than750; CoffeeSetupMug already has a native600-control horizon.

The retained panel contains126 full episodes from logical workers0–3, with
0 optimizer updates. Their original750/Coffee600 horizons were preserved
while measuring one-based first-success times, excluding reset controls.
The human decision stopped the remaining measurement after the second wave.

| Task | Episodes | Ever successes | Empirical first-success p99 | Success at original end | Success at600 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenDrawer |26|15|279|14|14|
| PickPlaceCounterToCabinet |26|18|538|18|18|
| PickPlaceDrawerToCounter |24|11|538|11|10|
| TurnOffStove |24|2|317|0|0|
| CoffeeSetupMug |26|6|524|6|6|

P99 uses nearest rank, conditional on ever success, separately by task. Every
successful sample here has fewer than100 observations, so this empirical p99
is the sample maximum. Failures are censored, not assigned a success time.
This is a fixed-bank prefix, not a completed50-case/task or population-tail
estimate. The two stove successes were transient; its final-success p99 is
undefined.

All52 ever-successful episodes first succeeded before600. Terminal success
is a separate criterion: one DrawerToCounter episode is false exactly at600
and true again by750. Thus the saved prefixes have48 terminal successes at600
versus49 at the original end. This tradeoff is disclosed with the selected
cutoff. Both future PPO arms and before/after panels must share600, and Flora
calibration must use matching600-step videos/terminal labels. Existing pinned
runs and old calibration artifacts retain their original duration.

Evidence: [decision](CUTOFF-600.json), [per-task values, compact episode rows and
raw-result hashes](PARTIAL-RESULT.json), [postprocessor](summarize.py),
[scheduler closure](SCHEDULER-CLOSURE.json). Native GPU probe passed for both
allocation GPUs at source `c5d271b40ed911b1b08136f2e2d40202b78ebb33`.
Raw three-view videos, success-step arrays, cases and full worker results stay
on Baige under `/mowenjie/fluffy-bassoon-2/outputs/robocasa-pi05-success-timing-20261102-r3/`.

## Closure and cost

Baige `job-kpdi47srqc2m` is manually stopped; the sole pod ended13:00:36
Shanghai and the job ended13:00:44. All126 action journals are resolved,
charging90,600 controls, with four complete worker results. No simulator
actions were issued by the remaining logical workers. New-model startup after
the completed second wave may have begun and is included in the GPU bound.

The worker's shell did not publish its normal terminal-charge receipt under
scheduler termination; its original clock remains unfinished and unchanged.
The external scheduler end plus the original60-second/GPU allowance bounds
R3 at1,534GPU-s; R1 preflight cost726 and unallocated R2 cost0 give a
cumulative upper bound2,260/9,600GPU-s. Unused budget is sealed.

The source-pinned clock metadata says236,000 actions due to a textual clock
replacement error. Actual per-rank escrows and the driver enforce the original
212,000 ceiling, and observed90,600 remains below both. Main's launcher
metadata is corrected after closure; the deployed source and raw clock are
retained unchanged. No further test suite or measurement is scheduled.
