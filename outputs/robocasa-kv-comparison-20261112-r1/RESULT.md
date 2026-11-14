# Historical four-transformer headline

All four original transformers were trained and evaluated: 12,000 audited episodes on three ID tasks (Drawer, Toaster, Coffee) and two held-out tasks (Fridge, Microwave).

| Method | ID success | OOD success |
| --- | ---: | ---: |
| Original SFT | 46.7% | 44.0% |
| Fixed PPO130 | 51.3% | 50.0% |
| AD | 46.7% | 44.0% |
| DPT | 47.3% | 49.0% |
| DIT | **55.3%** | **56.0%** |
| IC-IQL | 48.0% | 44.0% |

These are sixth-episode success rates; meta-learners use full history. DIT was +8.7pp ID/+12.0pp OOD above SFT and +4.0pp ID/+6.0pp OOD above fixed PPO. The ID PPO controls are the corresponding source-task teachers; the OOD PPO control is the predesignated OpenDrawer teacher. These baseline gaps are descriptive. [Baseline values and provenance](BASELINE-HEADLINE-20261114.json).

DIT with history reset scored50.7% ID/50.0% OOD. DIT led this particular panel, but its gain over resetting its own history was +4.7pp ID (95% interval −2.0 to 11.3) and +6.0pp OOD (−3.0 to 15.0). This is a positive observed history effect that still needs confirmation; the intervals include zero. There was one fit per method, unequal training durations and first-query replay variation.

This is historical evidence only. The original 130-update histories used the earlier PPO collector with the critic-target mismatch; they are excluded from the corrected, equal-budget three-versus-six-task comparison. The bug does not by itself establish why DIT led the old panel.

Per the November 14 retention instruction, keep this headline available and move closed payloads to verified cold storage. [Retention and restore record](../robocasa-six-task-scale-20261113-r1/STORAGE-REDUCTION.json) · [Audited numerical result](comparison/RESULT.json) · [Full original interpretation at its preserved commit](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/c096a7264c3f6a1d61505b637e3d19a3c56cce76/outputs/robocasa-kv-comparison-20261112-r1/RESULT.md).
