# Native 90° rotation execution

All times below are Shanghai time on 23 November 2026. The calibration pool contains nominal and rotated Panda interactions for each of the five adaptation seeds. The policy-driven collection and the three test arms use the same pinned LeRobot π₀.₅ weights. [Registered protocol and paper targets](REFERENCE-PROTOCOL.md) · [Reference implementation and CPU verification](REFERENCE.md) · [Frozen reader qualification](MATCHED.md)

| Baige Distributed Training job | Stage | Resources and status |
| --- | --- | --- |
| `job-jp5o2em8cal1` | `babble01` | Two Pro6K GPUs, started 15:05. The device probe passed. The simulator import failed before the first shard because the system library search path omitted the directory containing `libpcre.so.3`. The runner recorded 228 GPU-seconds. |
| `job-e6p4oaqs1u9n` | `babble02` | Four RTX 4090 GPUs, submitted 15:10. Changing the scheduler queue cleared its storage mount. The container log and job details confirmed the missing source and the absence of a mount. It exited before entering the launcher. The scheduler recorded 31 seconds of running time, or 124 GPU-seconds counting all four cards. |
| `job-rmvtf5i2znts` | `babble03` | Four RTX 4090 GPUs, submitted 15:16. Source filesystem `pfs-xPQ6Bb:/mowenjie` mounted read/write at `/mowenjie`. Library path repaired. The runner wrote the reservation on shared storage and its four-device probe passed. [Submission](evidence/submission-babble03.jpg) |
| `job-3ohhwfrwt34h` | `collect01` | Four RTX 4090 GPUs, submitted 15:18. Same verified filesystem and library path, four environments per policy lane. The launcher wrote its shared-storage receipt. [Submission](evidence/submission-collect01.jpg) |

The latest calibration jobs use detached Git pin `23a6641562a4d833837f27ab0535a20ad94e8ec8` on the shared run volume. Each has a 90-minute scheduler cap, 5,340-second launcher cap, and publishes sealed units for resumption. A single RTX 4090 development qualification produced a finite native action chunk after strict weight loading and a real NVIDIA-rendered simulator reset, then stepped the simulator. The development machine is reserved for that qualification. [Runtime records](RUNTIME.md)
