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
Target **500 new cases per task**, ten times the old count. Fix the final count
using outcome-blind throughput measurements and the remaining budget before
examining confirmation outcomes; do not stop when an interval becomes significant.

Full-history DIT executes all six physical episodes. After native qualification,
the history-independent controls execute only episode six with its original
noise stream. This gives nine physical episodes per case, or 22,500 at the target
count. No skipped support episode is fabricated. The endpoint is native
sixth-episode `success_once`, with no deployment optimizer updates.

Report per-task success, the three paired DIT contrasts, discordant-pair counts,
and separate equally weighted ID/OOD panels with paired 95% intervals. Resample
case identifiers within tasks, keeping all arm outcomes together. Preserve
failed-case evidence and disclose incomplete pairs. A clear history effect and
an advantage over the baselines support different claims; report them separately.
No numerical continue/kill threshold has been agreed. More cases narrow
uncertainty for this checkpoint; they do not establish training-seed robustness
or explain why DIT performs better.

Keep the existing **230 GPU-hour hard cap**. Reconciled conservative accounting
is 67 closed + 40 reserved for active teachers + 2 for paired qualification =
109 GPU-hours. Of the remaining 121, the proposal allocates up to 28 to the large
paired panel and 93 to corrected labels, eight fits and their evaluations.
The full paired reservation is conditional on timing and final reconciliation.
The old 50/170-hour caps remain closed.

Restore only required inference weights from verified archives. Keep complete
query/action/native-truth evidence and frame digests, with videos limited to
qualification or a declared sample; bank completed evidence in verified cold
storage and bound local staging. Do not restore the old raw training corpora.

**Current state (November 14, 16:22 Shanghai):** paired qualification
`job-rndlh3s50x3d` is running at `9f8bb6a4`. Its allocated GPU probe passed.
The first OpenDrawer wave completed all four arms plus the six-episode reset
replay audit. Its 1.29 GB evidence archive passed member hashes and full cold
readback. Four task waves remain; this is qualification, not the expanded
confirmation result. The first task's four-arm rollout took about 532 seconds
for eight cases, excluding setup, audit and archiving. The 500-case target is
conditional: use the complete qualification timing to fix a feasible sample
count within 28 GPU-hours before confirmation outcomes, as specified above.

All six corrected source collections, canonical packs and step-30 teachers
are complete. After correcting our launcher probe-purpose flag, two-task
labels `job-3r4f69s37tpv` passed its GPU probe and advanced; four-task labels
`job-972qpi5ao378` is queued. The failed label launch used 29 scheduler seconds
on two GPUs, conservatively closed at 0.2 GPU-hours; the retry has a shortened
174-minute limit, preserving the total 18 GPU-hour label allowance.
Conservative closed plus reserved usage is **127/230 GPU-hours**. The eight
new transformer fits and expanded paired confirmation have not launched.
Live accounting is in the [campaign approval ledger](../robocasa-six-task-scale-20261113-r1/APPROVAL.json).

[Execution and failure evidence](EXECUTION.json) · [Verified restored inputs](RESTORED.json) ·
[Machine-readable plan](PLAN.json) · [Core validation](CORE-VALIDATION.json) ·
[Budget reconciliation](../robocasa-six-task-scale-20261113-r1/evidence/BUDGET-RECONCILED-20261114.json)
