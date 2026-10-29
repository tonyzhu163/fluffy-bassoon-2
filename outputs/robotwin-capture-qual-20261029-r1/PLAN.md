# Camera collection parity and throughput gate

2026-10-29 Asia/Shanghai. This gate addresses the measured12.7actions/second
serial collection bottleneck in the real six-task experiment. It is not a
shortened learning recipe or a new scientific arm. No qualification actions
receive source-training credit, and existing r4 recoveries remain protected.

Candidate: keep native collect_obs=True sparse-reward/count logic, replace
intermediate camera reads with placeholders that retain native _update_render
side effects, then collect one final camera observation without an extra
render/RNG update. Use an opt-in flag restricted to demo_clean. Keep default
reference behavior. The native count is mandatory; no issued-length fallback.

Run focused Linux tests, then two place_phone_stand source scenes, seeds
615661050 and638361497. A fixed13x32x32 standard-Gaussian noise bank is shared
by full-reference A/B repeats and the candidate. Full repeats must match whole
traces. Candidate decision observations/features, decoded actions, exact
executed counts, rewards and done flags must match. Terminal observations are
reported separately and also gate admission. Both modes must pass native
sampler parity and unchanged/same frozen-motor fingerprint checks. Save raw
per-chunk hashes and outcomes. Publish a speedup only if all checks pass.

Resources: one Baige GPU,16CPUs,128GiB RAM,8GiB shared memory,8-minute
scheduler cap; tests90seconds maximum and driver330seconds plus20second
termination grace. Combined ceiling: r1reserve2m+r2duration14m49s+
r3duration2m49s+r4duration16m51s+this8mcap = **44m29s**, below45GPU-minutes.
No new training allocation or larger continuation is implied. The launch
receipt records the exact reviewed source commit and job identity.
