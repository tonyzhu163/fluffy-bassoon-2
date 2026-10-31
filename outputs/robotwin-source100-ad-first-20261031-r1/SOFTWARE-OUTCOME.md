# Allocated software checks: components passed, wrappers failed

Authoritative owner closure: c67b35c; compact receipts retained under
`outputs/robotwin-ad-first-software-qual-20261031-r2/`.

- AD7401:108 CPU tests passed, zero failure/error/skip.
- Teacher/source compatibility:57 passed,3 named legacy real-ledger tests
  deselected. This is software compatibility, not robot teacher competence.
- Actual source100 stage59588 and closed-tip20c agree on actor parameters,
  private RNG bytes and source/task/update counters; no CUDA RNG restored.
- No new bank production, student fit, robot episode or GPU time.

Both scheduler jobs remain Failed/exit1. Do not relabel the wrapper green.

1. R1 used absolute staged paths for pytest deselection, while node IDs were
   relative. Three intended exclusions ran and hit the retained import
   firewall. The fix sets an explicit rootdir and relative selectors; R2
   records the intended57pass/3deselected. No new AD failure was suppressed.
2. R2 completed its phases and owned cleanup, then terminal JSON publication
   failed: the run-ID string `run` was shadowed by helper `def run(...)`, so
   the terminal dict contained a function. c67 renames the helper `run_phase`.
   This is a minimal static fix, not a rerun or E2E Success qualification.

The reporting bug is isolated to the software-profile branch; normal bank
and live-parent execution use a separate path. Passed component receipts and
source outputs are not invalidated. Final100 bank admission and new sampled
AD500 readout qualification remain required in their own phase.

Actual costs: job-2wg5ue74mpid69s +job-d7v2vejcs5qk70s =139 CPU-worker-s,
1112 CPU-core-s, zero GPU-s. The600s software envelope is closed;461 unused
seconds are sealed, not retry capacity. No retry is needed merely to retain
component passes. The pending7200s combined phase includes139 once and leaves
at most7061 future CPU-worker-s only after its human decision. No production
bank/GPU/fit/readout authority is inferred from these results.
