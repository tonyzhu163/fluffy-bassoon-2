# Historical documentation

For the current research question and benchmark, read the [research design](../../docs/PAPER.md). The [findings](../../docs/FINDINGS.md) link measured outcomes to their run records. The [coordination board](../coordination.md) carries owners’ dated execution reports.

Retired proposals, launch handoffs and checkpoint-by-checkpoint status notes can be recovered at their original paths in the [pre-cleanup repository snapshot](https://github.com/tonyzhu163/fluffy-bassoon-2/tree/409baf7aca4c52959da72d55edd38c5f028b65c9). This includes the [constructed-history paper draft](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/409baf7aca4c52959da72d55edd38c5f028b65c9/fable.md), the [run directories](https://github.com/tonyzhu163/fluffy-bassoon-2/tree/409baf7aca4c52959da72d55edd38c5f028b65c9/outputs) and the [earlier literature reviews](https://github.com/tonyzhu163/fluffy-bassoon-2/tree/409baf7aca4c52959da72d55edd38c5f028b65c9/docs/literature). To read an individual file locally, use `git show 409baf7a:path/to/file`.

## Artifact retention

The [repository retention policy](../../README.md#artifact-retention) covers run evidence and checkpoints. Dated plans retained beside results document the conditions of those experiments. Some are referenced by launch records, evidence manifests or open handoffs. When reproducing an old run, use its recorded commit for code and relative paths. Earlier [coordination snapshots](coordination/) reflect the layout at the time of each report. For archived payloads, use the run's restore inventory.

Three historical reports also link payloads distributed outside Git. The [Astra artifact hashes](../astra-robocasa-demo-20261112-r1/artifact-sha256.txt) identify its replay, final views and dataset. The [Flora scheduler record](../robocasa-flora-rubric-failure-sample-20261102-r1/SCHEDULER-TERMINAL.json) records the scored job, and the [storage census manifest](../storage-audit-baige-20261107-r1/MANIFEST.json) identifies its bulk evidence.
