# Saved SAC65 CPU compatibility attempt

The exact `29df647fd30469fc04743baf47ba331718c9d1a2` gate exited before its first work phase. Baige job `job-eahrz09v6kkk` was a real 0-GPU, 8-CPU, 64-GiB allocation. It waited 120.012 seconds for an owner-issued actual resource proof and exited 1 when the proof did not arrive. Context compaction delayed the owner dispatch; the ticket reached shared storage at 04:53:38 Shanghai, after process termination.

No fixtures, inventory, projection, conversion, model imports, SGD, robot actions or VLA queries ran. This attempt supplies no evidence about converter correctness. Runtime output and late ticket are retained unchanged; no retry was launched.

The scheduler reports 149 running seconds (04:51:08 to 04:53:37), or 1,192 CPU core-seconds and zero GPU-seconds. The pod reports 144 seconds; those are separate clocks. Of the original 1,800 scheduler-second CPU envelope, at most 1,651 seconds remain for a reviewed diagnosed recovery. No fresh 1,800-second allowance is inferred.

[Terminal accounting](receipts/scheduler-accounting.json), [raw log](receipts/parent.log), [terminal receipt](receipts/terminal.json), [UI evidence](receipts/terminal.png).
