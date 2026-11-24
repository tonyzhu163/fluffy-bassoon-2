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
