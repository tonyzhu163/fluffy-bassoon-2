# Baige execution route for the command-frame screen

**Use Baige Distributed Training in Beijing, queue `算法Z组` (`aihcq-rh0wwk9muei4`), with one worker and zero GPU cards.** The authenticated create form accepted that queue and displayed 1 worker, 0 GPUs, 4 CPU cores and 16 GiB RAM. A previous job on this pool independently measured zero visible GPUs before masking, 2 CPU cores and 16 GiB in its container. The form and the historical allocation establish a route. Admission and runtime compatibility for this run remain pending. [Training console](https://console.bce.baidu.com/aihc/task/create) · [Previous Pro6K form](../archive/frozen-vla-readout-software-qual-20261101-r1/cpu-form.jpg) · [In-container allocation receipt](../archive/frozen-vla-readout-software-qual-20261101-r1/evidence/zero-gpu-attestation.json)

## Live capacity

The authenticated queue selector showed these figures on November 20, 2026, at approximately 14:26 Shanghai. CPU and RAM are queue-wide **free / maximum usable**. They do not guarantee placement on one node.

| Queue | Enabled workload | CPU cores | RAM (GiB) | Queued tasks |
| --- | --- | ---: | ---: | ---: |
| `算法Z组`, Pro6K | Distributed Training | 1,608 / 9,100 | 17,713 / 93,598 | 12 |
| `算法Z组-4090-dlc/在线任务`, `aihcq-chqrelndaaj4` | Distributed Training, online service | 121 / 1,944 | 1,088 / 16,182 | 21 |
| `通用-CPU`, `aihcq-olgqb2hl56tw` | Development machine, online service | 68 / 1,037 | 129 / 3,941 | 4 |

The CPU pool is `aihc-2axma32mmj40`. Its general queue had 17 nodes and 0 accelerator cards. Baige lists no Distributed Training workload for this queue. For the bulk job, open **分布式训练 → 创建任务 → 资源配置 → 队列**, select **算法Z组**, leave the accelerator type unselected and the count at **0**, and check the worker resource summary. The 4090 queue also displayed a zero-card draft, but this check located a prior zero-card allocation only for Pro6K.

## Existing Linux allocation for tests and qualification

The account's **开发机 → 只看我的** list showed `mwj-dangerous-dogs` (`dv-63739b1202`) running under `mowenjie` on `通用-CPU`. The existing **Notebook → JupyterLab terminal** answered a read-only probe at **14:28:04 Shanghai**. Linux reported host `aibox-r855e239307b-6bfb688cdd-lz2dx`, cgroup `cpu.max = 800000 100000` (8 cores) and `memory.max = 34359738368` (32 GiB). The project mount at `/mowenjie` reported 419 GiB available out of 2.0 TiB. This is filesystem availability. A per-project quota and a write probe remain to be checked when staging. [Existing development machine](https://console.bce.baidu.com/aihc/developmentMachine/webide?devId=dv-63739b1202&devName=mwj-dangerous-dogs)

Under `/mowenjie/fluffy-bassoon-2/setups/libero-meta-learning-20261117-r1/`, the simulator's `env/bin/python`, `simulator-source` and `libero-config` were readable. The release at `/mowenjie/fluffy-bassoon-2/sources/libero-pro2-e5500c69` returned HEAD `e5500c690c1498eca1b7050350654964190cdf70`. The Mesa EGL vendor file was present at `/usr/share/glvnd/egl_vendor.d/50_mesa.json`. These checks and the earlier successful four-worker Mesa replay support using this machine for CPU tests and short Panda/Sawyer qualification. The registered command-frame collector requires camera-free observations. If an import initializes EGL, the prior CPU launcher used Mesa software EGL with device 0. [CPU launcher](../libero-xemb-tcp-20261120-r1/run.sh) · [Prior Linux qualification and renderer failure](../libero-xemb-tcp-20261120-r1/EXECUTION.md)

## Release to execution

The inspected local base was `0fbc5e3167b362bc33b741e6bbfa20724aa93234`. After the owner verifies the implementation commit, deploy that exact SHA in a detached checkout on the project mount. Recheck the pin and writable run directory from Linux, then run CPU tests and the registered eight training-cell pulse checks. Count those tapes toward the collection budget. Before bulk collection, recheck queue capacity, the submitted zero-card request, the project mount and actual in-container allocation. A failed reset, signed response or inverse check stops collection. [Run plan](PLAN.md) · [Registered screen](../embodiment-meta-factory-20261120-r1/SCALE.md)

This was a route inspection. No scheduler job or reservation was created, no simulator or target was opened, and the worker used **0 GPUh**.
