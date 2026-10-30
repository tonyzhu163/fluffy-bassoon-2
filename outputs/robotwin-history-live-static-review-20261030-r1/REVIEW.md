# Bounded static review of the live target panel

Reviewed executable `2fd2096906e52aabeaeb984f333fe1aaa6c4e196` and launch artifact
`29003d206acb61cb8ecf483d989d103ecdac5d3e`. The earlier `66aff53` differs only
by the now-corrected full3200-action pre-runtime allowance check and its test.
This review performed no implementation, tests, ledger construction, GPU work,
process stop or changes to the running checkout.

## Current launch

The actual command pins the reviewed plan SHA
`b911dd972cc30e6f6b02cacf831eafa84738bd66ebd032824bb5fc94d340287c`.
It uses a5300-second parent wall bound,5330-second outer timeout with10-second
kill grace, and a separate90-minute scheduler allocation. The physical ledger
has a57,600 total/target ceiling and no source allowance. Eighteen serial fresh
K8 children cover nine declared conditions on two development tasks.

Static accounting and ordinary failure paths are consistent: only `run_trial`
reserves/resolves physical actions; reset/parity declares zero policy actions;
each K8 needs its full3200 allowance before loading an agent or runtime; native
units establish the first accepted seed/observation/feature banks; later units
must match those exact banks. Completed ordered receipts are rehashed and
reused without new charges. A failed/unknown child halts rather than retries.
Incomplete physical intents remain conservatively charged. SIGTERM/SIGINT,
normal child exits and native child aborts unwind cleanup of the child's owned
PGID/SID, including surviving descendants in that process group.

No new blocker was found on the actual hard-pinned normal launch path. The
execution owner reported its allocated22-test gate passed and the first native
K8 closed1,851 actions with no pending/uncertain work. Those are owner reports,
not an independent scheduler/runtime audit by this review.

## Limits handed to the owner

1. **Supervisor fatal loss needs allocation-level cleanup.** At
   `robotwin_history_live.py:114–122`, cleanup is a Python `finally`. A SIGKILL
   or fatal abort of the supervisor bypasses it; the fresh-session K8 child can
   outlive that parent until allocation teardown. Ordinary SIGTERM/INT and a
   native child abort are handled. The outer shell artifact alone does not
   prove immediate whole-container teardown. The owner acknowledged this exact
   limitation, retained the independent90-minute resource cap, and is preparing
   future parent-death guarding without changing the active pin. Descendants
   that create another process group are likewise outside `live_group`'s
   current group scan; the reviewed native child does not deliberately do so.

2. **The public plan-hash override is a future API hardening gap.** The parser
   accepts arbitrary `--plan-sha`; admission verifies the supplied checksum,
   rather than requiring the `PLAN_SHA` constant. Another compatible100-update
   endpoint plan could be substituted through that API. The actual launch
   explicitly supplies the original reviewed checksum, so this did not change
   its panel. A future pin should reject overrides or require explicit plan
   migration; the owner/exporter acknowledged it.

3. **Resume accounting needs actual scheduler charges.** The wrapper checks
   supplied prior seconds against its saved driver-time lower bound. That does
   not independently account for allocation setup or failed attempts. The
   execution owner must use actual cumulative scheduler spending and constrain
   the next outer allocation to the remaining90-minute allowance. The current
   fresh invocation uses prior0 and its own hard90-minute scheduler limit.

The source owner, exporter and coordinator received these findings. They
acknowledged the live parent was healthy, no altered endpoint plan was used,
and no silent migration/restart/hotpatch would occur. This bounded review is
closed; follow-up implementation and actual evaluation remain with those owners.
