# BOS deletion shortlist

November 14, 2026 Shanghai time. The seven repeats removed on November 13 already accounted for 3.02 GiB. This review read BOS and checked research dependencies. It made no further deletions. [Previous cleanup receipt](../bos-repeats-20261113-r1/RELEASE.json)

| Priority | What to remove | BOS savings | Condition |
| --- | --- | ---: | --- |
| 1 | Raw video, tensors and compressed traces from the RoboCasa initial B8 and final ID qualification panels | 2.87 GiB | Retain manifests, audit and case results. Removing the raw payload gives up exact episode replay. |
| 2 | Failed OpenCabinet PPO attempt01 tar | 8.55 GiB gross | Preserve the log, resolved config, accounting and manifests first. The job exited before a PPO update. Removing the tar gives up the exact failed-attempt initializer. |
| 3 | Full resume payloads in 59 intermediate checkpoints from the November 4 five-task PPO campaign | About 353 GiB net | Extract and verify each model weight against its original boundary, publish a readback-verified replacement and update the recovery pointers before deleting its tar. Rerun a historical likelihood check on the new format before the first release. |

[Exact BOS paths and measured bytes](BOS-CANDIDATES.json) · [Qualification and failure evidence](../outputs-retention-20261113-r1/BOS-AUDIT.md)

A read-only pilot verified the whole OpenDrawer step-60 tar, extracted its model export into temporary scratch and matched it to both the original boundary and the historical GPU-load SHA-256. That checkpoint alone would save 5.99 GiB. [Pilot readback](PILOT-READBACK.json)

The 59 intermediate tars occupy 734.91 GiB. Their uncompressed model exports total 381.85 GiB. The subtraction excludes minor metadata and is a storage estimate, with exact savings measurable after replacement. The full-weights exports are the payloads used by evaluation and historical likelihood replay. It gives up exact training continuation at those intermediate steps. The shortlist retains all ten step-zero and step-one qualification boundaries and six additional full snapshots referenced by the historical critic replay and checkpoint-sweep recovery receipts. The five final step-130 model files are still present on PFS. [Historical campaign closeout](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md) · [Critic replay](../robocasa-flow-sde-critic-parity-20261104-r1/RESULT.md) · [Checkpoint-sweep recovery audit](../robocasa-retention-audit-20261105-r1/RESULT.md)

The completed KV comparison still depends on about 321 GiB of raw archives for OpenDrawer, CloseToasterOvenDoor and CoffeeSetupMug, and DPT used their step-130 policies for labels. PickPlaceDrawerToCounter and TurnOffStove account for another 199 GiB and support the historical numerical audits. Keep all five corpora in this cleanup. [KV corpus provenance](../robocasa-three-task-kv-velocity-meta-20261105-r1/migrations/pfs-members-01/CACHE-MANIFEST.json) · [Five-task numerical audit](../robocasa-flow-sde-matched-denoise-drift-20261104-r1/RESTORED-Q1.json)

The new matched three-to-six-task plan explicitly excludes the buggy-critic corpora. Its verified corrected Drawer, Coffee and Counter archives already use about 360 GiB and are inputs for the next comparison. Preserve those, the verified Faucet reuse inputs, and any recoverable Toaster artifacts. The latest Stove attempt failed its job-local scratch check before training. BOS object deletion does not increase Baige PFS or scheduler-local scratch capacity. [Corrected comparison plan](../robocasa-six-task-scale-20261113-r1/PLAN.md) · [Latest collection status](../robocasa-six-task-scale-20261113-r1/EXECUTION.json)
