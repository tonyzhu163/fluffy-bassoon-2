# Three-teacher RoboCasa365 LoRA pilot

Both frozen-π0.5 students finished 576 optimizer updates. The memoryless LoRA student improved on frozen π0.5 and had lower held-out teacher-action error than DPT + LoRA on each task. For DPT, the matching prior SFT attempt had a small pooled advantage over another scene’s same-task attempt. Its interval included zero.

| Held-out flow-matching MSE, lower is better | Frozen π0.5 | Memoryless LoRA | DPT, own support | DPT, same-task donor |
| --- | ---: | ---: | ---: | ---: |
| OpenDrawer130 | 0.08855 | 0.08465 | 0.08845 | 0.08798 |
| CloseToasterOvenDoor130 | 0.09550 | 0.09335 | 0.09626 | 0.09849 |
| TurnOnSinkFaucet40 | 0.05933 | 0.05862 | 0.06150 | 0.06138 |
| Pooled | 0.08113 | 0.07887 | 0.08207 | 0.08262 |

The 576 held-out queries cover 48 reset scenes, 16 per task. Both students used the same 1,728 training queries, frozen motor, flow draws and minibatches. All evaluation conditions reused the same noise and timestep on each query. We averaged the 12 chunks within each reset and bootstrapped resets within task for 10,000 draws. The paired reduction from frozen to memoryless was 0.00225 (descriptive 95% interval 0.00145 to 0.00313). The DPT model had 0.00320 higher error than memoryless (0.00214 to 0.00438). Swapping the matching support for a same-task donor increased pooled DPT error by 0.00055 (−0.00082 to 0.00201). The intervals condition on the single fit per arm. Student simulator success remains unmeasured. [Plan and data contract](PLAN.md) · [Full precision](metrics.json) · [Paired query errors](held-out.json)

Source rewards and success labels came from native RoboCasa simulation. Targets came from the final actor’s actual executed controls on its saved query observations. Every support paired the earlier SFT attempt on the same initial scene with that query episode. All 192 scene XML identities were distinct, and none of the pilot seeds overlapped the separate 512-case teacher tests. The memoryless student trained 139,264 parameters. The history-conditioned student trained 673,920 including its GRU and projection.

Baige job `job-2hjwe0bl6k1x` completed successfully on one Pro6K queue GPU at 19:46:01 Shanghai on November 10. Both adapters exactly reproduced the unmodified model at initialization, then received nonzero gradients. Batch size six peaked at 7.59 GiB allocated. The job ran for 230 seconds (0.064 GPU-hours), or 258 seconds (0.072 GPU-hours) including scheduler startup. Training commit `9bcd917bbdac7b601fff543c4e4d4228f60dabea`. The combined adapters, optimizer, sampler coordinates and RNG checkpoint remain on Baige under this run’s `fit-01/checkpoint.pt`. [Scheduler receipt](completed.jpg)
