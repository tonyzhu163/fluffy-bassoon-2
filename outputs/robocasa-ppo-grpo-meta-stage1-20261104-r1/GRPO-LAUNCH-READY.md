# GRPO launch contract

Submitted on November 4 to Baige Pro6K as `job-r4b89rtmuhbx`. The [in-job probe](GRPO-GPU-PROBE.json) passed on all eight devices. The [execution record](EXECUTION.json) tracks progress. The following settings describe the submitted job.

The human approved 64 GPU-hours total and explicitly requested GRPO alongside
root-cause analysis, matching executed action steps. No new approval is needed.

Use the existing Baige console and copy the completed PPO job
`job-bylh99ylx9ie`. Change the name to
`mwj_1104_stage1_grpo_OpenDrawer_r1`. Recheck current queue capacity, limits
and actual accelerator type before submitting. Preserve its Pro6K / 算法Z组
queue, one node, 8 allocated GPUs, 168 CPUs, 1280 GiB RAM, 32 GiB SHM,
240-minute maximum duration, automatic retries disabled and PFS mount.

Command:

```bash
TASK_META=/mowenjie/fluffy-bassoon-2/sources/stage1-grpo-013c2f10 TASK_CODE_PIN=013c2f10c84b22db834dfe533907bb8a01737768 bash /mowenjie/fluffy-bassoon-2/sources/stage1-grpo-013c2f10/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1/LAUNCH.sh grpo
```

The clean Git-bundle deployment records the pending publication commit above.
GitHub publication is currently blocked: the connector rejects blob creation
with403, local SSH authentication fails, and local HTTPS has no credential.
Do not describe this commit as pushed until verified.

The launcher explicitly retains two actor ranks and microbatch16; the full
sampling and actor-batch records are kept. Latest upstream changes since
46daecc9 affect other run bookkeeping/fit scripts only. The learner and core
checks are unchanged from the55-test Linux qualification.

Contract: OpenDrawer,128 episodes/update,16 scenes ×8 attempts,40updates,
one optimizer epoch,6Adam steps/update/rank,3,072,000TRAIN controls,
192,000DEV controls at0/10/20/30/40,noise.3,ignore_last=true,no critic,
4,100,000global action escrow. Final raw costs include native reset controls.
The new output identity is `/mowenjie/fluffy-bassoon-2/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1/grpo/attempt-01`; it must not exist before launch.

PPO is terminal, with conservative17.1267GPUh consumed. This GRPO arm reserves
at most32GPUh, leaving14.8733GPUh below the64GPUh all-attempt hard ceiling.
At the last check PFS had493GiB free; a footprint comparable to PPO leaves
over300GiB. Keep the200GiB checkpoint-exit guard. The human's parallel request
removes the earlier archive-before-GRPO scheduling preference. Never reclaim
any active job or another owner's files.

After submission, record its actual job ID; require the in-job8-device probe,
weight/normalization hashes, exact grouped reset identities and advancing
updates, and monitor every20minutes. Bulk PPO archival remains pending,
with final resumablePPO40 retained.
