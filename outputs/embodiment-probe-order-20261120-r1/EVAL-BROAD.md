# Broadfit01 source development

**Broadfit01 step 6400 scored 23/50 Panda (46%) and 16/50 Sawyer (32%).** Both arms fell below the registered 38/50 threshold. The candidate stops at development. Recounting all 100 native reward tapes reproduced the aggregate report exactly. [Native evidence](evidence/broad/broad-native.json)

Baige Beijing Distributed Training job `job-alhror6iloul` succeeded on 2026-11-20 at 16:28:33 Shanghai. It used one RTX PRO 6000D (RPBZZZ6), eight simulator workers and stride five. The first progress check at 16:22:41 found 24/50 paired layouts prepared. By 16:25:01 all 50 had passed preparation and 12 Panda receipts were present. All three stages exited zero. Submission to success took 10m07s, within the 95-minute limit. The live scheduler check before submission found the completed fit and no active matching evaluation. [Scheduler](evidence/broad/scheduler.json) · [Progress](evidence/broad/broad-progress.jsonl) · [Submitted command](evidence/broad/broad-launch.json)

| Checkpoint on the same 50 layouts | Panda | Sawyer |
| --- | ---: | ---: |
| **Broadfit01, step 6400** | **23** | **16** |
| Categorical parent, step 6400 | 14 | 15 |
| Extended regression, step 6400 | 34 | 20 |
| Original regression, step 1600 | 25 | 6 |
| Recovery fit, step 2400 | 17 | 15 |
| Waypoint fit, step 5200 | 1 | 1 |

Against its categorical parent, broadfit gained 18 Panda starts and lost 9. On Sawyer it gained 11 and lost 10. Broader fitting data and additional optimization changed together, so these differences measure the continued-fit checkpoint as a whole. [Paired layout IDs and prior report hashes](evidence/broad/broad-native.json)

The fit at `29f1db5581f1a674002cebff2ef723342e8a6604` continued from the verified categorical parent with fresh AdamW. The selected `spatial_bins_v1` checkpoint has SHA-256 `4720a1910cd6fe6c2592a75ea92f782955bac5ea77f3125b791c0ea215c740ad`. All 31 learned tensors changed and both normalization tensors matched. The clean hostrecover scorer at `e2f794c41c9141a7aadaf67339da919336362f22` reproduced the registered donor bytes and four geometry fingerprints at every seed from 740000 through 740049. Zero of 50 layouts overlapped either fit’s exposure or the 50 released donors. The child fit contained 95 distinct generated fitting layouts. [Model fingerprints](evidence/broad/broad-model.json) · [Panel and exposure audit](evidence/broad/broad-exposure.json)

| Verified SHA-256 | Digest |
| --- | --- |
| Fit manifest | `6d4fc40b60249fb19e09525890b315fa5dcb29cf4098ada459c1577ec2322f53` |
| Pinned run.sh | `8a259ccbed50e6f7cbce3800512cf0142648b998179704df955c52f57cf09169` |
| Host-local evaluation manifest | `6492662090c28905153f55814d54063f0bbbc3e788dad1678c54bd83e1fe2143` |
| Shared registered geometry | `3b7e9e274586ae622a37d9b5fa33381833d37d1e0e673965096653dd5fa71798` |
| Native aggregate report | `149513cb3e514ec908ddb9500fe1bdc7a1ce97ada893ff0637ff0227d7a42cc0` |
