# Closed PPO pilot traces archived

All 256 immutable train07 simulator trace files were copied byte-for-byte to the existing BOS archive mount. Every file passed a full independent cold readback, a complete staged restore with an independent SHA256 check, an unchanged-original check and a visible-reader check before its original was released. This reclaimed 46,744,842,538 bytes (43.53453 GiB), with peak staging 187,723,418 bytes (0.17483 GiB). Staging was removed after completion. No GPU work ran.

The archive ran from Git commit cc0324a29163197b376b90311acb29320837b7cb; the deployed script SHA256 is e597bb18e7a7c1d6c2e4b087fd05c5206ef942dc586734f76e65e43e7d9fe8aa. Its dependency review read the actual immutable scientific source 48fa7ea4: both active arms collect fresh traces, their final actor restore reads their own global_step_2, and the post-training reward readout reads their own canonical histories. The closed pilot traces have no current/final consumer.

All 36 train07 checkpoint/boundary/worker files retained their original inventory stamps. Closed canonical metadata/restore pointers, shared caches, qual02 fixture and both active science outputs were retained. No checkpoint or tensor reserialization occurred.

The cold root is /mnt/bos/adgfyl/strong-plant-archives/robotwin-sidney-matched-baselines-20261106-r1/train07-traces-archive01. Each original path has an adjacent .pt.gz.cold.json pointer. [STATUS.json](STATUS.json) records every original path, byte count, SHA256, cold copy, verification result and exact restore command. [Launch provenance](../train07-traces-archive01.launch.json) records the CPU process and deployed source.

At 07:05:26 Shanghai, the pair was still collecting its second iteration: both checkpoints contained 20 updates per rank; all 192 common initial scene groups matched. Shared free space was 452.390625 GiB versus the remaining-forward gate of 403.549247 GiB. This retention release creates margin for the coordinator's separate resolution of the other evaluator's fixed 411 GiB guard; it does not change that evaluator or any active contract.
