# Saved-PPO r1: behavior replay gate blocked optimization

Baige job-p0zpb6ugu2pm, source6a97e9b653b7d2ce69f63cf6edb838aa6a3f20fb.
Driver exited1 after25.31seconds; scheduler stopped at its one-minute limit
with61seconds recorded, terminal2026-10-30 02:02:47 Shanghai. The allocation
probe passed. The likelihood/value replay gate rejected the unchanged saved
policy **before any optimizer step**. Retained updates0, new robot actions0.
The immutable original input SHA remains722618543005ffcac5591593f67f2f8eaa3bd9fcc1be5cebf66995e9487306a9.

Static diagnosis found a missing global numerical setting: the pinned motor's
RLinf/rlinf/models/embodiment/openpi/pi0.py:169 calls
torch.set_float32_matmul_precision("high"). The standalone feature-only update
had not constructed that motor and therefore omitted the setting. R2 explicitly
restores it and records precision plus replay errors before its gate. This is
a supported diagnosis pending the corrected GPU replay, not a relaxed tolerance
or an update result. R1 did not retain the numerical error magnitudes; R2 fixes
that diagnostic gap. The main three-arm recipe and original artifacts are
unchanged.

The earlier2minute startup reserve was audited against Baige's actual49second
receipt (startup-actual-49s.png), releasing71unused seconds. Reconciled actual
usage after this attempt is43m10s, leaving110seconds inside45minutes. One fresh
corrected one-minute allocation is therefore proposed within the original cap;
no two-hour training approval is implied.
