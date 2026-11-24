# Generated plate teacher collection

The source archive contains successful scripted Panda and Sawyer demonstrations on 253 shared generated BASE Goal 5 layouts. The collector locks their order by SHA-256 of the ASCII string `plate-transfer-v1/` followed by the scene fingerprint. Ranks 0–15 go to training and ranks 16–19 to held-out diagnostics. If a native source replay fails, its entire pair drops, the failed slot takes the next unused rank starting at 20 and both arms replay. Input corruption and runtime failures terminate the attempt. No target outcomes enter selection or replacement.

Each replay starts from the archived sampler state and uses the exact issued command prefix through native termination. It records both camera streams, measured XYZ, quaternion, individual finger joints, commands, reward and the four-transition public history. The reset must match the physical scene, task, model and generated row. Terminal reward must be one. All earlier rewards must be zero. A replay may reach success earlier than its archived counterpart.

Per layout, training takes 12 full windows and held-out diagnostics take eight. The first two phase slots spread ages 0–3 across layouts. Later slots span the trajectory through its last ten-command window. The mixed variant replaces alternating phase slots with Sawyer, rotating the assignment across layouts. The two training manifests share row-to-layout and row-to-phase assignments. Both fits have common fixed diagnostics covering all eight held-out windows per layout per arm. The original pilot's SourceDataset reads the packed bytes back against digests formed from native tape rows before atomic publication.

The parent submits one exclusive Baige Distributed Training RTX 4090 with a two-hour allocation. The collector uses one simulator process and at most two Linux cores. The separate two-core Mesa smoke uses the first matched pair and writes to its own attempt. No stock policy is loaded for teacher replay.

## Deployment and CPU qualification

The collector runs at `dec6b1f3d6f33e134e5fcb75185a99f761845216` in a detached checkout made from a Git bundle. The native wrapper is pinned to `a8a997e7d06bb131d3f9710204c58e106f0f0904`; the task release is `e5500c690c1498eca1b7050350654964190cdf70`. Baige CPU cores 26 and 27 ran `cpu01` to exit 0 in 73 seconds. Panda and Sawyer reached native reward one at 98 and 82 issued commands on rank 0, seed 301090. The smoke used Mesa llvmpipe and packed a twelve-window mixed layout. [Native receipts](evidence/cpu01/enrollment.json) · [Archive selection and reserve order](evidence/source-order.json)

The SHA-256 of the ordered 253-candidate file is `ce1d4da211f9fe76512dcefa19c0c645cffc6b59ac6f86ad163d44110cd96dc0`. The parent received the GPU launch command at 13:53 Shanghai time on November 24. The intended invocation from the pinned checkout is below.

```bash
bash outputs/frozen-vla-plate-transfer-20261124-r1/data/launch.sh gpu dec6b1f3d6f33e134e5fcb75185a99f761845216 gpu01
```

## Distributed collection

Baige Distributed Training job `job-h3fk587i4jg2` ran on one exclusive NVIDIA GeForce RTX 4090. The scheduler allocated 12 CPUs and 64 GiB RAM. The launcher restricted computation to two cores and used one EGL renderer. A CUDA arithmetic, UUID, PCI and process check passed on the allocated device. All native receipts identify its NVIDIA 4090 renderer. The PFS mount reported 113,724,358,656 bytes available at launch. Full replay tapes used 782 MiB and the final datasets used 101 MiB as reported by `du -sh`. [GPU probe](evidence/gpu01/gpu.json) · [Storage readback](evidence/gpu01/storage.txt)

The launcher started at 13:54:10 Shanghai time on November 24. The first two native positive episodes were verified by 13:56:56. At 13:59:42 all 16 training pairs were complete. The job published the complete dataset, including the four held-out pairs, and exited 0 at 14:00:27. Launcher time was 377 seconds (0.10472 GPU-hours). The scheduler's 120-minute limit was an upper bound. [Launch receipt](evidence/gpu01/launch.txt) · [Native output](evidence/gpu01/console.log) · [Exit receipt](evidence/gpu01/exit.txt)

An independent readback using `ce9ca43a58a4f4a99d9f37d3626f39903f871e68` ran on Baige development CPU cores 26 and 27. It took 11 seconds and exited 0 at 14:00:42. It verified the seal emitted by the native launcher, all tape and HDF5 digests, source-only enrollment order, shared row bytes, balanced body counts and disjoint placements. The trainer and parent received the seal via the shared handoff. The original source-order digest was identical in the CPU smoke and GPU collection. [Readback metrics](evidence/readback01/readback.json) · [Readback pin and invocation receipt](evidence/readback01/launch.txt) · [Readback exit](evidence/readback01/exit.txt)
