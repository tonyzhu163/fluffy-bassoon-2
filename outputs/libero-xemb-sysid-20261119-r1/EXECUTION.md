# Execution on Baige

On November 19 the project volume had 478 GiB free of 2 TiB and 2.46 million free inodes. A write/read succeeded. The allocation exposes no separate quota utility. The released simulator is pinned to e5500c69 and OpenPI to 215abfb2. The frozen checkpoint path is recorded in the motor source. The scheduler budget for this workstream is 12 GPU-hours across attempts. CPU and software-rendered qualification are excluded from that total.

| Attempt | Code | Measured outcome |
| --- | --- | --- |
| Development CPU, cpu01 and cpusim01 | 75b607cd | Stopped on missing pytest and a mixed null/string gripper registry before any model result. |
| Development CPU, cpu02 and cpusim02 | 6a93ea8b | Three tests passed. Five native probe episodes on two reset groups completed, the reader-facing frame-zero image and state matched within each group, and the four reader fits ran for 20 smoke-test updates. |
| Development CPU, cpu03 and simqual01 | 95388e6e | Three tests passed. The updated motor interface issued all 15 probe controls in a real shifted simulator, checked the schema and returned native reward and termination fields. |
| Distributed Training, job-94hoq9rcbqeo | 95388e6e | Submitted 19:56:59 Shanghai, one NVIDIA RPBZZZ6, 12 CPU cores, 48 GiB RAM, 90-minute wall limit and zero automatic retries. Queue displayed 27/416 GPUs free, personal headroom 11/32. The worker passed the GPU probe and rendered with NVIDIA EGL. |

The first distributed job runs smoke and then Gate 1 on eight development tasks. Each arm uses the same physical start, reset seed and motor noise for all six interfaces on that task. The gate uses the full 48-pair denominator, including BASE. The experiment compares a public coordinate convention to the frozen motor's training convention. The simulator camera geometry and gripper friction stay at their released values in this run. See the separate [friction experiment](../libero-xemb-contact-20261119-r1/RESULT.md).
