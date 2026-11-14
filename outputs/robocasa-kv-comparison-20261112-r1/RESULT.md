# Historical four-transformer headline

All four original transformers were trained and evaluated: 12,000 audited episodes on three ID tasks (Drawer, Toaster, Coffee) and two held-out tasks (Fridge, Microwave).

| Method | ID success | OOD success |
| --- | ---: | ---: |
| AD | 46.7% | 44.0% |
| DPT | 47.3% | 49.0% |
| DIT | **55.3%** | **56.0%** |
| IC-IQL | 48.0% | 44.0% |

These are sixth-episode full-history success rates. DIT led this particular panel, but its gain over resetting its own history was +4.7pp ID (95% interval −2.0 to 11.3) and +6.0pp OOD (−3.0 to 15.0). Reliable adaptation from history was not established. There was one fit per method, unequal training durations and first-query replay variation.

This is historical evidence only. The original 130-update histories used the earlier PPO collector with the critic-target mismatch; they are excluded from the corrected, equal-budget three-versus-six-task comparison. The bug does not by itself establish why DIT led the old panel.

Per the November 14 retention instruction, keep this headline available and move closed payloads to verified cold storage. [Retention and restore record](../robocasa-six-task-scale-20261113-r1/STORAGE-REDUCTION.json) · [Audited numerical result](comparison/RESULT.json) · [Full original interpretation at its preserved commit](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/c096a7264c3f6a1d61505b637e3d19a3c56cce76/outputs/robocasa-kv-comparison-20261112-r1/RESULT.md).
