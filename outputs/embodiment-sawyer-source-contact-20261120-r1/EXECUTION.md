# Native Sawyer replay on Baige

All times are November 20, 2026, Shanghai time. Baige Distributed Training admitted **job-6xb480p72ol9** at **16:47:00** under the name `mwj_1120_sawyer_contact_native01`. The job list showed Running shortly afterward.

| Setting | Verified value |
| --- | --- |
| Allocation | One worker, zero GPU cards, 8 CPUs, 32 GiB RAM, 4 GiB shared memory, 60-minute scheduler limit. The command has a 59-minute timeout and automatic retries are off. |
| Scheduler | Beijing Pro6K queue 算法Z组, `aihcq-rh0wwk9muei4`. The completed zero-card probe-order job `job-owxei8dm5dvv` supplied the template. Before submission the search for `sawyer_contact` returned zero jobs. The queue showed 2,338 free CPU cores, 25,305 GiB free RAM and one queued task. Individual CPU and memory quotas displayed as unspecified. |
| Staging | Existing authenticated SSH access to the Baige project mount and a separate existing console tab. Write/readback succeeded with 399 GiB and 1,565,706 inodes available. Neither quota utility was installed on the staging host. |
| Diagnostic | Detached source `bf4f0195ef8ab0a5708dab2d1f3f790da0945dd7`. Git bundle SHA-256 `d161fd95ecddeec0d53cba8827161c45199b6c14f20007c19d0a840bea3978f5`. Bundle, clean detached checkout and individual script hashes verified before submission. |
| Native evaluator | Original detached checkout `fc61bf2afb48d4639664db93dae64ae9cb04e2df`, original native simulator interpreter, LIBERO-Pro2.0 `e5500c690c1498eca1b7050350654964190cdf70`. The camera-free R2 session takes recorded issued actions and returns native reward and proprioception. |
| Original cohort | Numerically audited Huber source scorer `job-9zpwf7aapj3u`, `large-panel-v3/dev`, stride 5 at update 6400, Sawyer seeds 740000–740049. Full report hash `7e2122efbd976638680c8215c3830eeb974179c5944f5185b38771988a3d464a`, full checkpoint hash `77d18c386b2cc176252cfb68449e4415748382f554046ca50367ed4592676d9f`. |

The launch command invokes `run.sh bf4f0195ef8ab0a5708dab2d1f3f790da0945dd7 native01` from this run's detached checkout. The launcher permits only a Linux scheduler worker, checks its CPU limit and absence of accelerator device nodes, and runs six independent episode processes at most. All new PFS outputs are under this run's directory.

At 16:47:11 the worker verified the native Python 3.11.14 interpreter, 8.0 cgroup CPUs, 32 GiB memory and zero NVIDIA device nodes. At the first progress check 12/50 episodes had validated, with zero rejections or exceptions.

At 16:49:02 the launcher exited 0. All 50 cases validated. The worker independently reconstructed contact masks and first-touch progress from the saved numeric traces, then wrote the 202,045-byte compact 50-case JSON. The scheduler reported Succeeded at 16:49:24. Its displayed duration was 2m11s, with 2m24s from creation to terminal timestamp. This was the sole submitted job and used zero GPU-hours. [Worker log](evidence/native01/run.log) · [Allocation](evidence/native01/allocation.txt) · [Exit](evidence/native01/exit.txt) · [50-case result](evidence/native01/replay/cases.json)

Copied back all 50 per-case records and 50 compressed traces, the two aggregate JSON files, launch and exit receipts, source hashes, worker log and allocation receipt. All 107 files passed local SHA-256 readback. The checksum manifest itself hashed to `5a067c2aee7d548acd880925b9960746d2ffdd746559c35027d1f1fdeca5a4bd` on both hosts. The compact result hashed to `d785f2a0beaa22195fdc74853c77c04ff62ff7611f85140fc2daa51eef0e6309`. The four returned source digests matched the pinned local scripts. The PFS copy is under `/mowenjie/fluffy-bassoon-2/outputs/embodiment-sawyer-source-contact-20261120-r1/native01/`. [Manifest](evidence/native01/SHA256SUMS) · [Source digests](evidence/native01/source.sha256) · [Result](RESULT.md)

The final task-list refresh returned one matching job with status Succeeded, owner mowenjie, 1 worker, 0 GPU cards, 8 CPUs and 32 GiB RAM. [Scheduler readback](evidence/scheduler.json)
