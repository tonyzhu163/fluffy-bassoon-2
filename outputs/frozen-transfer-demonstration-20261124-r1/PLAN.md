# Source demonstration approach with frozen motor handoff

Registered at 2026-11-24T03:58:29.395034+00:00, before inspecting prospective endpoint evidence. Repository base 311b4f8a24ace9602f8bb5dfc818155489484287. This is the final fixed study.

## Question and prediction

Can one task-specific source IIWA demonstration provide the approach for the Kinova3 target and leave grasp and placing to the tuned frozen π0.5 motor? The primary contrast pairs tape 34 with standard tuned control. Tuned failures that become native successes support rescue on these starts. No rescues, or paired losses offsetting rescues, count against a net benefit. Tape 35 is a separate, development-selected secondary comparison. Raw control provides the unadapted comparison. Prior work selected 35 on four Kinova development rows. Whole-tape successes (35 on 37 and 34 on 14) motivated this test and do not establish handoff performance.

## Fixed protocol

Use official LIBERO-Pro2.0 libero_goal task 3, nominal BASE, Kinova3. Use the eight exact released initial layouts 40 through 47 with seeds 42710300 + layout row. Call them reused layouts because a separate CPU-only fixed-tape study already uses them. At registration, the target raw, tuned and hybrid endpoints have not been observed. Never use row ID or target observations to choose a donor or adjust the method. Use the original instruction and the shared cue study's 15-control separate layout 11 PR7 seed 41 calibration for the standard tuned Perceiver. Freeze all weights. Use one π0.5, matching starts and global-replan noise across four modes on each row (32 complete native episodes).

1. Raw π0.5.
2. Standard tuned Perceiver.
3. Tuned with raw archived IIWA tape 34 approach. Primary donor, shortest successful source, chosen using source evidence alone.
4. Tuned with raw archived IIWA tape 35 approach. Secondary donor, chosen earlier using four target development rows.

For each donor compute the cut exclusively from source commands as five times floor(first zero-based positive grip command index / five), after verifying open/close semantics and initial open command. Tape 35 expects 115. Issue exactly the archived already-issued seven-channel actions in order through cut minus one, bypassing all corrections for these controls. Hand off before any positive source close. At cut and every five controls thereafter replan on current target observations with original instruction, corresponding global replan noise and ordinary tuned action conversion. Target proposals during prefix are permitted for matched compute. Only the native environment defines success. Stop on native success or after 300 controls.

## Qualification, comparison and stopping rule

Audit the true outgoing step, matched starts and noise, first fresh observation and real tuned transformation at handoff. Use Baige CPU slots 38 and 39 after checking overlap, and representative native fixtures for each donor extending beyond its cut. Exercise actual motor proposals and tuned conversion at the boundary as required for the launch. Keep both fixtures and their limitations explicit. Use the standard probe, pinned detached checkout and native runtime for the dedicated one-4090 distribution attempt, budget under one hour. Parent owns Git, UI, submission and monitoring. No endpoint inspection before registration and no design revisions in response to any outcome.

Report paired absolute native successes, rescues, losses and net differences against tuned and raw by donor with uncertainty and individual matched outcomes. Treat rows as a fixed reused panel. Distinguish whole-tape CPU validation from within-attempt matched evidence. Keep compact portable native evidence, receipts, configs and a result note here. No further studies.
