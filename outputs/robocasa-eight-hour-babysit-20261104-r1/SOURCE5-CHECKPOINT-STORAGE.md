# Source checkpoint storage qualification

The user requested storage optimization before launching the approved source campaign. No active run or vendor checkout was modified.

A real closed two-rank four-epoch PPO checkpoint contains 16,134,571,318 bytes: 9,177,023,284 bytes of complete distributed recovery plus 6,957,548,034 bytes of standalone inference weights. The distributed checkpoint already contains all model parameters, optimizer, LR scheduler and RNG. Reconstructing all675model tensors on allocated Linux CPUs matched the standalone export exactly.

For the new sources, retain every distributed recovery checkpoint and every actor/rollout/environment sidecar. Export standalone full weights at final40 only. The normal boundaries1/10/20/30 retain complete recovery; final40 additionally retains the full inference export needed by fixed-endpoint TEST and DPT relabeling. DEV evaluation uses the live actor. The default remains unchanged for other configurations, and export schedules missing the final teacher are rejected.

This saves 43.12% at each intermediate actor checkpoint and 27,830,192,136 bytes (about25.92GiB) across four intermediate boundaries. Total actor-checkpoint storage falls from 75.13 to 49.21GiB. Nothing is quantized, and no optimizer/sampler/RNG state or chronological source chunk is dropped.

The former170GiB per-source reservation is reduced conservatively to150GiB after measured25.92GiB export savings, retaining additional margin. The shared200GiB floor remains unchanged; require the200GiB floor plus150GiB per new source, the remaining reserved growth of active sources, and20GiB headroom before allocation and again inside the job. Sources may overlap only when these independent reservations fit; otherwise they are serialized. A separate base-plus-trainable-delta format is not introduced here.

Validation:24Linux CPU core tests pass, including exact next-sample/Adam-step recovery without an inference export and retaining the final export. The actual FlowActor override passed the upstream-writer API check. Both task configs retain every scientific value of the earlier qualified recipe:40updates,4epochs,3.072MTRAIN controls,960Adam steps,global minibatch256/microbatch16,H50/10flow steps,noise.3,ignore_last=true,unchanged critic/rates/seeds. Only checkpoint export layout changes.

Evidence: SOURCE5-CHECKPOINT-DCP-QUALIFICATION.json and SOURCE5-READY-PREFLIGHT-01.json. Qualified code6308a9f83cc307c9f94d57c8ab86323cb9f1f86a. CPU-only qualification used0GPU-hours.
