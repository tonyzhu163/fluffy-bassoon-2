# Independent DPT/full ID/OOD evaluation

This phase evaluates only DPT/full, whose own validation plateau was reached at23800 updates. It never reads AD or DPT/static readiness. The checkpoint and data/runtime contracts are frozen in binding.json. Driver e8c7641611899805612e17cbb97be31a871a51f3; native motor/features/bounds/controller core remains68f2c5aa33218709453a7821f6c6223ac3803035.

Use the generic READOUT.py/COMMAND.sh with TASK_READOUT_METHODS=["dpt_full"], TASK_READOUT_FRESH_ID=1 and this phase's distinct absolute output directory. Binding and plan checks ran on allocated Baige Linux CPUs with CUDA hidden. The saved checks demonstrate admission while DPT/static remains unconverged.

Evaluate the identical locked ID50 OpenDrawer and OOD50 CloseFridge cases, reset and motor draws used by the AD-first phase. DPT/full gets K0, one full support episode with actual terminal feedback, then a separate same-case K1 query with frozen weights. Report support separately. This phase has300 physical episodes/285900 conservative controls; no new source or PPO training. Native horizons remain750/900. Compare against initial/final PPO bank from the AD-first phase after verifying exact case/noise/initial-feature/base/source hashes. Publish this model's own ID/OOD results independently; only the final combined comparison waits.

Baige job job-t4xwnjnjuxp9, attempt dpt-full-r1, was submitted13:30:27 Shanghai on2026-11-02. It is queued for4x2 GPUs,32CPU/128GiB RAM/16GiB shared memory per node,479-minute operational chunks and no automatic retry. At submission only2 GPUs were free, so scheduler availability is the current gate. Preserve all completed/partial records and diagnose failures before fresh attempts. Release its GPUs on its own completion. Monitor every20 minutes with the existing experiment heartbeat.

**Hold:** job-t4xwnjnjuxp9 was manually terminated13:35:04 Shanghai after the AD phase exposed repeated-reset visual pairing failure. It briefly acquired GPUs and produced GPU manifests, but no policy episodes. Preserve that startup allocation and its costs. Requalify the cause before a fresh attempt.
