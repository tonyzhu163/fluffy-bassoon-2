# Native Sawyer contact fork on Baige

All 50 Sawyer development layouts completed on the first attempt. The raw motor succeeded on 20, the fixed expert taking over after first hand contact on 31, and the same expert starting from reset on 37. The takeover rescued 18 of 30 raw failures and lost 7 of 20 raw successes, a net gain of 11 layouts or 22 percentage points. All 150 native branch receipts passed. [50 paired cases](evidence/native01/compact/paired-cases.csv) · [Full result and provenance](evidence/native01/fork/cases.json)

| Branch | Native successes | Sum of native returns | Issued controls |
| --- | ---: | ---: | ---: |
| Raw Huber motor tape | 20/50 | 20 | 11,497 |
| Raw prefix, then fixed expert | 31/50 | 31 | 9,170 |
| Fixed expert from reset | 37/50 | 37 | 7,253 |

## Branch identity and controls

All three branches had byte-identical opening state snapshots on each layout. The raw and takeover branches also had identical snapshots after the first native contact control and identical issued-action and reward prefixes. Handoffs followed controls 36 through 82, with median 48. The expert issued the next control and inherited the remaining portion of the 300-control horizon. All 50 raw replays reproduced the archived control sequence, replan checks, native rewards, termination and prior geometry traces. No layouts were rejected or missing. Of the 19 layouts the takeover failed, the same expert solved 11 from reset. [Receipt audit and outcome cross-tabulation](evidence/native01/compact/receipt-audit.json)

A second readback on Linux checked all 50 compressed capture hashes, compared the saved opening and handoff arrays byte for byte, and checked the recorded action source, step index, native reward, terminal flags and receipt for every control. The compact archive holds all 27,920 issued 7-D actions with their native signals. Each member is named `<seed>__<branch>` and stores one row per control with fields `actions`, `action_source`, `native_steps`, `rewards`, `success`, `terminated` and `truncated`. Action source 0 means the archived motor and 1 the fixed expert. [Control and reward archive](evidence/native01/compact/controls-rewards.npz) · [Array audit](evidence/native01/compact/capture-audit.json) · [Per-branch timings](evidence/native01/compact/timings.json)

## Job and runtime

Baige Beijing Distributed Training job `job-q8jarvv0p7wn` used the successful zero-card template `job-6xb480p72ol9`. The search for owned jobs matching `sawyer_contact_fork` returned zero before submission and one succeeded job at closeout. The Pro6K queue `aihcq-rh0wwk9muei4` showed 2,236 CPUs and 24,067 GiB RAM available with no queued tasks at admission. The job used one worker, eight CPUs, 32 GiB RAM, 4 GiB shared memory and zero GPU cards. Automatic fault tolerance was off. The scheduler limit was 60 minutes, with a 59-minute command timeout. The worker measured an 8.0-core cgroup limit and zero NVIDIA card nodes. [Scheduler readback](evidence/SCHEDULER.json) · [Allocation](evidence/native01/allocation.txt)

| Event on November 20, 2026 | Shanghai time |
| --- | --- |
| Job created | 17:27:30 |
| Pinned launcher started | 17:27:46 |
| First three-branch layout validated and observed in the terminal | 17:28:19 |
| 27 of 50 validated with zero errors observed in the terminal | 17:29:49 |
| Launcher exited 0 after all 50 validated | 17:31:01 |
| Scheduler terminal status Succeeded | 17:31:19 |

The native program took 193.547 seconds of wall time. The scheduler displayed 3 minutes 30 seconds, and creation to terminal status was 3 minutes 49 seconds. This was the sole attempt and consumed zero GPU-hours. Seed 740000 ran as the native admission before the six-worker pool processed the other 49 seeds. [Launch](evidence/native01/launch.txt) · [Worker log](evidence/native01/run.log) · [Exit](evidence/native01/exit.txt)

## Source, data and storage

The parent supplied `096489dbd22d58ad9c6cf67b2a9ea463d0d15dfd` before submission. The mounted staging repo pointed to an older local bundle, and direct hosted Git from staging lacked a trusted host key. The Mac produced an incremental Git bundle against the preceding diagnostic pin `bf4f0195ef8ab0a5708dab2d1f3f790da0945dd7`. Authenticated SCP transferred it to the project PFS. The remote checked the bundle, detached revision, clean checkout, three script digests and native Python CLI import before submission. Bundle SHA-256 was `fbd91c605c5c1a537cf9b276c4647cf3d07c401dad625909f04b4886a4ea03b1`. [Deployment receipt](evidence/deploy-native01.txt) · [Job-side source digests](evidence/native01/source.sha256)

The native evaluator was `fc61bf2afb48d4639664db93dae64ae9cb04e2df` on LIBERO-Pro2.0 `e5500c690c1498eca1b7050350654964190cdf70`, using the preceding run's Python 3.11.14 interpreter and software EGL. The original Huber stride-5 checkpoint at update 6400, native development panel, and previous Sawyer contact replay were verified by SHA-256. The 50 source layouts were seeds 740000 through 740049. [Complete source and dataset identities](evidence/native01/fork/cases.json)

The SSH endpoint was verified against the preceding job's launch and exit receipts and the native evaluator and simulator pins. The same read-write `mowenjie` v1 dataset was mounted at `/mowenjie`. A fresh-file write/readback passed with 398 GiB and over 1.54 million inodes free. User quota was unavailable on the staging host. [Storage receipt](evidence/storage.txt)

The PFS run is `/mowenjie/fluffy-bassoon-2/outputs/embodiment-sawyer-contact-fork-20261120-r1/native01`. The 50 numeric captures total 14,722,231 bytes and remain there. The returned control/reward archive is 716,600 bytes. All 64 transferred files passed local SHA-256 readback. The file manifest matched on staging and on the Mac. [Transfer manifest](evidence/native01/SHA256SUMS) · [50 PFS capture digests](evidence/native01/compact/captures.sha256) · [150 control array digests](evidence/native01/compact/controls.sha256.json)

| Artifact | SHA-256 |
| --- | --- |
| Full 50-case result | `3e325074b21b06f5e6242c0eb48692d41312929c2003a924653b2627114ac606` |
| Compact controls and rewards | `ac380ace607427cce05cf328f3060713c3db8398537331be72cdc4268d4aafd5` |
| Transfer manifest | `5e6ad408d1b135c079e0c70426c76cf3eba899538dfb37d6382763dd0d3b926f` |
