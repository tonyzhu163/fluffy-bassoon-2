# Original LIBERO-PRO pilot monitor

All times are Shanghai (UTC+08:00) on 2026-11-23. Distributed Training job `job-9me3dekaeqaj` requested one Pro6K GPU for 90 minutes. The monitor reads the existing job's shared receipts over SSH. [Run contract](PROTOCOL.md)

| Time | Observation |
| --- | --- |
| 14:01:19 | Baige accepted the submission and initially reported it queued, according to the launch record. Submitted revision `0c457b8fadc2f63c3d212a830b70f6d440eac48b`. |
| 14:05:58 | The shared run directory was readable on `baige-dev`. The job wrapper had yet to create its `gpu01/` directory. |
