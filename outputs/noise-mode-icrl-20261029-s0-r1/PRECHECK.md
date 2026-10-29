# Stage 0 DSRL noise-arm tests

Source: `352edc407738ce14b6a0adb9100732326ca40902`, fetched from main on 2026-10-29.

Use the upstream Stage 0 code and specified collection shapes. Run the 3 focused Linux test modules first. The launcher adds the plan-required admission after S0a: require 68 complete rows, exact fixed-arm repeat trajectories, nonzero first-chunk arm effect, and projected total collection time within 6,900 seconds. If a check fails, retain evidence and stop before S0b/S0c. No learner training. Scheduler maximum: one GPU, 16 CPU, 64 GiB RAM, 8 GiB SHM, two hours.

Mac AST and Bash syntax checks passed. School aliases are unavailable on this Mac. The configured Baige development endpoint is reachable; model computation and tests run only in the scheduler allocation.
