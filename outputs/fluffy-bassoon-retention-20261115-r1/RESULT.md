# Baige fluffy-bassoon archive and output census

On November 15, 2026 Shanghai time, 13 files moved from `/mowenjie/fluffy-bassoon/outputs` to verified BOS recovery. The batch released **108.23 GiB** of allocated PFS space. Shared PFS free space rose from **22.94 GiB to 131.04 GiB**.

| Released payload | PFS GiB | Evidence kept on PFS |
| --- | ---: | --- |
| Nine MIKASA Table 10 step-1 and step-2 qualification weight files | 98.12 | Qualification ledgers and checkpoint metadata, initializers, and all three reported production recovery checkpoints |
| Four completed MIKASA carry-download archive pieces | 10.11 | Original download and restore receipts, plus the verified restored carry policy under `inputs/` |

Every BOS object passed full SHA-256 readback and an independent copy back into temporary scratch. Each released PFS file has a recovery pointer with the original path and hash. The active job view showed five LIBERO jobs and zero owned active workflows. No jobs were stopped.

[Per-object verification](VERIFICATION.json) · [Restore instructions](RESTORE.md) · [Scheduler sample](SCHEDULER.json)

## Output footprint and archive order

The pre-release output census counted 228 run directories. A whole-tree scan counted **506.68 GiB** once per hard-linked inode. The run-attributed sizes below count each inode once in each run directory, so a shared initializer appears in each corresponding row.

| Run | Pre-release GiB | Archive decision |
| --- | ---: | --- |
| MIKASA Table 10, November 14 r1 | 164.17 | Released 98.12 GiB of qualification weights. Next candidate is the cancelled campaign’s complete production recoveries together with the November 15 retry recovery. |
| LIBERO K=2 priorities r3 / r2 | 69.30 / 68.43 | Keep current policy inputs available. Ten full online policy exports across r2/r3 occupy about 109 GiB. Check the evaluation and profiling readers and verify compact, exact replacements before releasing closed export copies. |
| MIKASA T1 flow / recurrent flow | 37.87 / 32.72 | The matching seed initializers already share three inodes across both runs. Retain them for slim checkpoint loading and review only independent superseded copies. |
| ICLR rebuttal November 6 | 35.21 | Review retired diagnostics and staged copies within the directory. Its CTM initializer is named by newer MIKASA launch contracts. |
| MIKASA Table 7 r1 / r2 / r3 | 21.97 / 22.16 / 22.63 | Archive older r1/r2 handoff payloads after validating their recovery dependencies. Retain the r3 step-10,000 checkpoint and formal paired-cue evidence. |
| LIBERO adapted K=1 T1 r2 | 22.48 | An owned LIBERO job was active. Leave its inputs and outputs on PFS. |
| Bridge ticks | 14.54 | Four initializer paths already share one inode. Keep the baseline input and paired tick evidence. |
| MIKASA Table 10 November 15 r2 | 11.14 | Review alongside the other two cancelled production boundaries. Combined, their three full-weight files occupy 32.71 GiB before counting training state. |
| MIKASA carry input restore staging | 10.11 | Released all four download pieces. The restored policy remains on PFS. |

[Full run inventory](INVENTORY.jsonl) · [Exact release selection](PLAN.json)
