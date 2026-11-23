# Kinova3 native evaluation

The frozen π0.5 policy completed **2/6** layouts with untreated Kinova3 images, **3/6** with a uniform Panda source colour and **3/6** with per-link Panda source colours. The two recolourings rescued different moka-pot layouts. Across the six layouts, the simple uniform control matched the linkwise method. Every entry below is a native BASE LIBERO-Pro2.0 episode from an unchanged Kinova3 action interface. [Paired counts and differences](evidence/gpu01/result.json)

| Task / initialization | Raw | Uniform source colour | Per-link source colours |
|---|---|---|---|
| Drawer and bowl / 14 | Failure, 300 | Failure, 300 | Failure, 300 |
| Drawer and bowl / 15 | Failure, 300 | Failure, 300 | Failure, 300 |
| Two moka pots / 14 | Failure, 520 | Failure, 520 | Success, 505 |
| Two moka pots / 15 | Failure, 520 | Success, 475 | Failure, 520 |
| Cream cheese and bowl / 14 | Success, 94 | Success, 93 | Success, 91 |
| Cream cheese and bowl / 15 | Success, 99 | Success, 95 | Success, 95 |

Counts after each outcome are native 20 Hz controls. Both variants changed the hash of the first frozen action chunk on every layout. Opening wrist images, public proprioception, simulator physics and flow-noise banks were matched on all six. Both rescues occurred on the task with a 520-control horizon. The per-link and uniform variants each gained one outcome and lost one against the other. Their zero net difference at each task makes the three-task bootstrap degenerate. Against the within-job raw condition each gain is 1/6, with a descriptive task-cluster bootstrap interval from 0 to +39.2 percentage points on three purposively selected tasks. [Eighteen complete native reward records](evidence/gpu01) · [Registered protocol](PLAN.md)

## Independent raw repeat

A parallel job on a second RTX 4090 reproduced **4/6** on untreated Kinova3. The two additional successes were precisely the moka-pot layouts, in 496 and 361 controls. All six complete reset records matched exactly across jobs, including both raw and preprocessed views, motor proprioception, instruction, initial physics and the complete pre-generated flow-noise bank. Both jobs loaded the same Git revision and frozen checkpoint path. Nevertheless the first action proposal hash differed on **6/6** starts. Four of the six native reward tapes and lengths matched bit for bit. The observed cross-job change of two raw successes exceeds either apparent one-success benefit from recolouring.

The independent native job evaluated each unmodified opening three times within the same policy process, in its raw, 3D-action-adapter and planar-action-adapter episodes. Their policy input is identical before the first command. All three opening hashes agreed for each of six layouts. The visual job compared different rendered external observations within one process and did not include a raw→raw replay. The jobs used two physically distinct 4090s. Their scripts requested different CPU thread counts (visual 2, native 1) and the native job used a headless OpenCV overlay. None of those configuration differences identifies the cause of the opening divergence. [Within-job control hashes](evidence/gpu01/raw-opening-controls.json) · [Visual launcher](launch.sh) · [Native environment](../frozen-vla-native-transfer-20261123-r1/env.sh)

The data establish sensitivity to the rendering intervention and variability in raw outcomes under replication. They give no reliable evidence that either adapter improves native success or that link-specific paint has a performance advantage. These six layouts and two jobs are insufficient to estimate a population variance or assign the discrepancy to a particular CUDA kernel. In the pinned OpenPI sampler, preprocessing runs in evaluation mode and an explicit noise array bypasses random noise generation. The saved hashes establish the divergence at or before the first frozen motor output. This run saved that output as a digest, so its numerical magnitude cannot be recovered from these tapes. A matched repeated-raw control within each independent implementation is warranted before spending on a learned appearance adapter. [Cross-job receipts with individual hashes](evidence/gpu01/raw-cross-run.json) · [Independent native experiment](../frozen-vla-native-transfer-20261123-r1/PLAN.md) · [Pinned OpenPI sampler](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L217-L230)

Baige Distributed Training job job-4orb6exianqe used one RTX 4090. Its worker completed all 18 native episodes at 18:42 Shanghai on November 23, exiting 0 after 682 seconds (0.189 GPU-hours of launcher wall time). Source commit a8a997e7d06bb131d3f9710204c58e106f0f0904. [Allocation](evidence/gpu01/gpu.json) · [Launch](evidence/gpu01/launch.txt) · [Exit](evidence/gpu01/exit.txt)

## Linux CPU qualification

The Panda source palette and the native Kinova3 intervention ran on Baige with Mesa software EGL using LIBERO-Pro2.0 revision e5500c690c1498eca1b7050350654964190cdf70. The inherited simulator and environment wrapper match repository commit 1068ed0c514152c7df59193555a81f28ed5d4759. The development scene was BASE Goal 4, initialization 16, seed 41810416. The parent sequence of native controls, complete simulator-state digest, source material descriptor, arm identities and object-layout digest are recorded.

| Opening comparison | External-view changed pixels / 65,536 | External-view mean absolute RGB change / 255 | Wrist-view changed pixels |
|---|---:|---:|---:|
| Uniform Panda colour vs unmodified Kinova3 | 1,824 | 0.234 | 0 |
| Per-link Panda colours vs unmodified Kinova3 | 1,542 | 0.323 | 0 |
| Per-link vs uniform | 1,825 | 0.163 | 0 |

The full simulator state at reset and after each of the two fixed native controls matched bit for bit. Every non-image public sensor, native reward and termination bit also matched. At those three states the two reconstructions changed 1,819–1,828 external-view pixels relative to each other. Both wrist images were identical to raw at all three states. This is a small appearance contrast. It establishes that the intervention reaches the camera input with unchanged measured physics on a development start. The native task evaluation above uses the same source palette. [CPU measurements](evidence/cpu.json) · [Source palette](evidence/palette.json) · [Source blob identities](evidence/blobs.txt) · [Simulation implementation](adapter.py)

The independent transport check executed four six-control CPU trajectories through the production stock motor loop, including a second replan after five controls. A fixed probe supplied the policy actions. Unmodified transport and the experiment’s raw transport matched all opening fields, physics, motor input/noise/first-proposal hashes, rewards and lengths. All three appearance conditions passed the paired-state checks. The GPU job supplied the frozen policy for the subsequent native evaluation. [Transport results](evidence/ipc.json) · [Implementation identities](evidence/ipc-blobs.txt)
