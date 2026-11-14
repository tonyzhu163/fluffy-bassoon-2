# Larger paired DIT evaluation

Recorded November 14, 2026, following the instruction to run many more paired
evaluations and quantify DIT's advantage over SFT, PPO and its own reset control.

The old panel has only 50 cases per task. DIT's observed history benefit is
positive, but its paired uncertainty intervals include zero. This is insufficient
to resolve the continue-or-kill decision confidently.

| Historical sixth-episode result | ID | OOD |
| --- | ---: | ---: |
| SFT success | 46.7% | 44.0% |
| Fixed PPO130 success | 51.3% | 50.0% |
| DIT with history reset | 50.7% | 50.0% |
| DIT with full history | 55.3% | 56.0% |
| DIT full minus SFT | +8.7pp | +12.0pp |
| DIT full minus PPO | +4.0pp | +6.0pp |
| DIT full minus reset, paired 95% interval | +4.7pp [−2.0, 11.3] | +6.0pp [−3.0, 15.0] |

[Verified values and sources](../robocasa-kv-comparison-20261112-r1/BASELINE-HEADLINE-20261114.json).
The SFT/PPO gaps above are descriptive; their paired intervals have not yet been
computed. ID uses each task's corresponding PPO130 teacher. OOD uses the
predesignated OpenDrawer PPO130 transfer teacher.

Evaluate four frozen arms on identical fresh cases and sixth-attempt noise:
DIT full history, DIT reset, original SFT and fixed PPO130. Keep the original
three ID tasks (Drawer, Toaster, Coffee) and two OOD tasks (Fridge, Microwave).
The initial conditional target was 500 new cases per task. Outcome-blind
throughput measurements fixed the actual count at **256 fresh cases per task**
before examining confirmation outcomes; do not stop when an interval becomes
significant. The fixed count and its resource rationale are recorded below.

Full-history DIT executes all six physical episodes. After native qualification,
the history-independent controls execute only episode six with its original
noise stream. This gives nine physical episodes per case, or **11,520 at the fixed
count**. No skipped support episode is fabricated. The endpoint is native
sixth-episode `success_once`, with no deployment optimizer updates.

Report per-task success, the three paired DIT contrasts, discordant-pair counts,
and separate equally weighted ID/OOD panels with paired 95% intervals. Resample
case identifiers within tasks, keeping all arm outcomes together. Preserve
failed-case evidence and disclose incomplete pairs. A clear history effect and
an advantage over the baselines support different claims; report them separately.
No numerical continue/kill threshold has been agreed. More cases narrow
uncertainty for this checkpoint; they do not establish training-seed robustness
or explain why DIT performs better.

The decision is to collect the larger paired panel before treating the old
intervals as evidence to continue or kill the direction. An interval spanning
zero does not establish no history benefit. Pairing uses the same physical case
and query-noise stream across arms; report the paired gains and losses, rather
than inferring a difference from separate success-rate intervals. More cases
can improve precision but do not guarantee a decisive result. A precise small
effect and an imprecise effect warrant different conclusions; a positive
baseline gap alone does not establish that history caused the gain. No new
practical-effect threshold is being introduced after observing results.

Keep this historical-checkpoint confirmation separate from the corrected
[AD/DPT/DIT/IC-IQL three-versus-six-task comparison](../robocasa-six-task-scale-20261113-r1/EVALUATION-MIGRATION.json).
That comparison uses eight fresh, matched-duration fits and 40 paired cases per
task per model. The 256-case confirmation does not increase its sample size.
Both studies report ID and OOD separately; OOD means held out from meta-training,
not necessarily unseen by the base VLA during pretraining or SFT.

Keep the existing **230 GPU-hour hard cap**. At the November 14, 17:52 check,
113 GPU-hours were closed conservatively, 28 reserved for historical
confirmation and 48 reserved for the two fit allocations: **189/230 committed**.
The remaining 41 comprise 32 for matched new-fit evaluation and 9 of uncommitted
recovery margin. The old 50/170 caps remain closed.
[Current ledger](../robocasa-six-task-scale-20261113-r1/APPROVAL.json).

The outcome-blind count is now **256 fresh cases per task**, fixed before any
confirmation outcomes: 1,280 cases and **11,520 physical episodes**. Full
qualification measured all five task waves, including audits and verified
archival. On four GPUs, 256 predicts 369.3 minutes; the next fully balanced count
of 288 predicts 414.9 minutes, beyond the 405-minute worker deadline. The
original conditional target of 500 predicts 48.4 GPU-hours, above this panel's
28-hour allowance. The scheduler limit is 420 minutes, with automatic retries
off. This is a throughput/budget choice, not a significance threshold.
[Fixed decision](DECISION.json) · [Timing evidence](evidence/QUALIFICATION-TIMING.json).

Restore only required inference weights from verified archives. Keep complete
query/action/native-truth evidence and frame digests; bank completed evidence in
verified cold storage and bound local staging. Do not restore old raw corpora.

**State at November 14, 17:14 Shanghai:** qualification
`job-rndlh3s50x3d` succeeded at source `9f8bb6a4`. All five task waves passed
native audit and complete cold readback (408 physical episodes including the
serial-reset check). Every recorded reset control replayed exactly from a fresh
episode controller; the eight OpenDrawer serial/direct reset pairs had no
outcome discordances. This validates the control procedure, not a DIT advantage.
Expanded confirmation `job-ir52nloomqdx` was submitted at17:18 Shanghai,
source88ebc32e, and passed its four-device allocation probe.
[Confirmation execution](CONFIRMATION-EXECUTION.json).

All six corrected source collections, step-30 teacher labels and corpus assembly
are complete. Corpus-query qualification passed, and both four-method
transformer allocations started. Live progress and the next check belong in
[coordination](../coordination.md).

[Execution](EXECUTION.json) · [Restored inputs](RESTORED.json) ·
[Machine-readable plan](PLAN.json) · [Core validation](CORE-VALIDATION.json)
