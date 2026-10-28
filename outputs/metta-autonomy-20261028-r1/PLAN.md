# Autonomous CTM/CMM benchmark campaign

Authorized objective: work autonomously for ten hours starting 2026-10-28
23:28 Asia/Shanghai (through 2026-10-29 09:28). Seek a reproducible advantage
in peak performance, held-out generalization, or training efficiency; measure
all three. Move to VLA steering after the benchmark evidence supports it.
Do not manufacture a positive result or redefine a qualification as a result.

Reviewed latest GitHub main 65b48a457f9cf1c9c9dcd1c9813ac9e53d5e0455.
Dedicated detached worktree incorporates the ML10 owner's qualified source
cd0128f without modifying its worktree or active runs. Existing ML10 jobs and
school CMM calibration retain ownership and immutable contracts. The required
proxy_on invocation is unavailable on this Mac; no proxy/routing settings are
changed. Git bundles through the configured Baige development endpoint deploy
commits. School aliases currently lack usable local configuration; retry only
through their documented aliases, never direct school IPs.

Additional campaign budget: one concurrent Baige GPU, at most ten allocated
GPU-hours total including qualification, with scheduler caps and actual usage
receipts. This does not add duplicate ML10 jobs. Initial qualification cap:
45 minutes, one RPBZZZ6 / 16 CPU / 64 GiB, eight simulator workers. Stage the
robotics package in a separate dependency directory; never mutate the active
ML10 Python environment. Run tests only in the allocated container.

Sequence:
1. Implement PointMaze and HalfCheetahVel with hidden tasks, explicit splits,
   real interaction counters and shared PPO/parallel environment execution.
   ML1 Push exists already. Qualify all four architecture cores and actual
   default-shaped updates before choosing run budgets.
2. Audit current CMM calibration where access permits. Treat CMM experiments
   as exploratory until calibration qualifies its implementation.
3. Use measured costs to screen PointMaze and HalfCheetah with CTM/CMM/GRU/XL,
   then devote remaining budget to confirmation seeds, independent trained
   reset controls and length/parameter generalization for promising contrasts.
   Preserve negative and capped runs. Inspect live ML10 results without
   interfering with its ongoing owner.
4. Audit raw arrays, source and checkpoint hashes; produce curves against real
   interactions and allocated GPU-hours. A single seed is a screen only.
5. Before VLA learning, require useful held-out in-context adaptation plus a
   repeatable advantage on at least one of the three axes. Otherwise diagnose
   the remaining failure, and report it honestly at the campaign deadline.

Check launched jobs every twenty minutes, sooner on failures or actionable
changes. Keep full optimizer/sampler/normalizer/RNG recovery state. Save raw
records under each immutable run and update only the single coordination file.
