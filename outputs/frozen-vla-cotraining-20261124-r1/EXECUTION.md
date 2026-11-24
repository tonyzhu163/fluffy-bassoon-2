# Baige co-training and native evaluation

Baige Distributed Training accepted `job-9v3iytmhh9ig` on November 24 at 11:35:10 Shanghai, on one RTX 4090, 12 CPU cores and 64 GiB, with a 90-minute allocation limit. The job's in-container training probe passed at 11:35:27. The trainer is pinned to `6785abc8323473b81973b9d69c93863bf523e4d0`; its source-only manifest hash is `05209732b5e3f04308c8acbef2f6c8c8a2f173673840fdb055e1b094ff7d6f40`. [Scheduler receipt](training-job.jpg) · [Source data](data/RESULT.md) · [Training and freeze verification](model/README.md)

The parent owns training and evaluation submissions. The native held-out-task and task-matched panels consume the same read-only fit. Each requires a completed export and verifies its identity before scoring.

Attempt 1 stopped before fitting after 94 seconds on one GPU. The on-device comparison found an absolute history-input gradient difference of 0.00000771 against a direct native gradient norm of 0.00010505, despite exact CPU agreement. No fitted bundle or test scores were emitted. The recovery uses this as an execution discrepancy to diagnose and preserves the source split and output contracts. [Attempt 1 evidence](evidence/training-attempt01.json)
