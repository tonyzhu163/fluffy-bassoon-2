# Checkpoint parent-path failure

Baige job-sgeu36o4ay2n, immutable source0531f509, failed2026-11-02
at17:47:41 Shanghai. All four workers exited and all GPUs were released.
All four actual GPU probes passed. Native resets passed the human-directed
physical/image-diagnostic policy. The first checkpoint callback failed because
its episodes/ parent directory had not been created. This was a runner repair
mistake; no alternative-interface rollout or PPO update occurred.

Four newly completed native episodes retain videos, truth, reset snapshots
and pre-dispatch case/motor/control plans. Their chunk records were not saved.
The fresh recovery audits these rows and reuses their valid outcomes, while
keeping this output and source unchanged. They are not complete AD/DPT histories.

All four process clocks close:493GPU-seconds. A conservative scheduler
creation-to-terminal upper bound is720GPU-seconds, bringing the two attempts
to3644 of43200. There were17450 all-in physical controls,0 unresolved calls,
and2400 policy controls in complete episodes. See TERMINAL-ACCOUNTING.json.

The shared atomic writer now creates missing parent directories while
preserving exclusive temporary-file writes. The29 Linux tests include actual
nested disk writes through the lane callback, payload round-trip, completed
sibling preservation and rejection of an unresolved partial file. The fresh
continuation is robocasa-interface-screen-20261102-r3.
