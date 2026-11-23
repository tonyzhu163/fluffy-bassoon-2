# Linux CPU qualification

The Panda source palette and the native Kinova3 intervention ran on Baige with Mesa software EGL using LIBERO-Pro2.0 revision e5500c690c1498eca1b7050350654964190cdf70. The inherited simulator and environment wrapper match repository commit 1068ed0c514152c7df59193555a81f28ed5d4759. The development scene was BASE Goal 4, initialization 16, seed 41810416. The parent sequence of native controls, complete simulator-state digest, source material descriptor, arm identities and object-layout digest are recorded.

| Opening comparison | External-view changed pixels / 65,536 | External-view mean absolute RGB change / 255 | Wrist-view changed pixels |
|---|---:|---:|---:|
| Uniform Panda colour vs unmodified Kinova3 | 1,824 | 0.234 | 0 |
| Per-link Panda colours vs unmodified Kinova3 | 1,542 | 0.323 | 0 |
| Per-link vs uniform | 1,825 | 0.163 | 0 |

The full simulator state at reset and after each of the two fixed native controls matched bit for bit. Every non-image public sensor, native reward and termination bit also matched. At those three states the two reconstructions changed 1,819–1,828 external-view pixels relative to each other. Both wrist images were identical to raw at all three states. This is a small appearance contrast. It establishes that the intervention reaches the camera input with unchanged measured physics on a development start. Native task completion under the frozen VLA requires the scheduler evaluation. [CPU measurements](evidence/cpu.json) · [Source palette](evidence/palette.json) · [Source blob identities](evidence/blobs.txt) · [Simulation implementation](adapter.py)

The independent transport check executed four six-control CPU trajectories through the production stock motor loop, including a second replan after five controls. A fixed probe supplied the policy actions. Unmodified transport and the experiment’s raw transport matched all opening fields, physics, motor input/noise/first-proposal hashes, rewards and lengths. All three appearance conditions passed the paired-state checks. The frozen weights and their native task success are tested only by the GPU job. [Transport results](evidence/ipc.json) · [Implementation identities](evidence/ipc-blobs.txt)
