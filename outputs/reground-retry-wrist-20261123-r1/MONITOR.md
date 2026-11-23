# Native wrist retry on Baige

Parent submitted Distributed Training job `job-iynjhzjijoyj` at 18:31 Shanghai on 2026-11-23 for four RTX 4090s. The [submission receipt](submission.jpg) shows it queued. All times below are Shanghai.

| Check | PFS evidence | Next action |
|---|---|---|
| 18:36:40 | The shared run mount was reachable through `baige-dev` with 234 GiB available. The `scored/` directory did not yet exist. No launcher, GPU or native batch receipt was available. | Watch for the first attempt receipt and independently validate the first published batch. Next scheduled status check 18:56. |
| 18:56:51 | A fresh SSH read of the shared mount still showed no `scored/`. There were zero published batches and no evidence from which to charge GPU time. The last scheduler evidence available here was the 18:31 queued screenshot. | Continue watching for startup or a parent-supplied terminal scheduler status. Next scheduled check 19:16. |
| 19:16:31 | PFS remained reachable and `scored/` was still absent 45 minutes after submission. No launch, failure, or scored output was visible. | Keep the event watch active. Next scheduled check 19:36. |
| 19:35:55 | Shared run directory reachable. No `scored/`, admission, rank receipts or native batches 65 minutes after submission. | Continue watching. Next scheduled check 19:56. |
| 19:55:56 | Still 0/70 publications with no launcher receipt. A Baige CPU read independently revalidated the four-lane qualification fixture at the original pin and joined the 1,400 historical labels. The historical seed-0 rates are 81/280 successful and 75/280 unsafe. [CPU receipt](evidence/readback-preflight.json) | Await the first production batch. Next scheduled check 20:16. |
| 20:16:38 | Fresh PFS check found no launcher directory or native batches. | Watch for startup. Next scheduled check 20:36. |
| 20:37:04 | Still no scored store. A CPU watcher on Baige is armed to run the pinned production validator as soon as the first batch is published. | Continue PFS checks. Next scheduled check 20:57. |
| 20:57:53 | The [PFS event log](evidence/monitor-events.jsonl) still showed no launch, GPU admission, rank receipts, batches or exit. | Continue on the 20-minute cadence. Next scheduled check 21:17. |
| 21:17:53 | No launcher, admission, batches or exit in the event log. The full CPU readback is armed to verify all 70 producer and contract pins before calling the producing checkout. | Monitor the first production receipt and continue 20-minute checks. Next 21:37. |
| 21:37:53 | No launcher, GPU or rank receipts, published batches or exit. | The [timestamped PFS log](evidence/monitor-events.jsonl) records subsequent 20-minute checks and earlier state changes. Next 21:57. |
| 21:57:53 | No launch or native publications. | Continue PFS watch. |
| 22:12:43–22:13:53 | Original pin launched on the expected job host. Exclusive GPU admission passed on four distinct RTX 4090s. | Check rank receipts and first batch. |
| 22:14:01 | Terminal exit 2. Ranks 0, 2 and 3 pointed at a missing driver under the reference transport directory. 0/70 batches and no partial work. | Wrote [ALERT](ALERT.md) and notified parent. Preserve receipts and qualify the replacement shell on CPU. |
| 22:19 | Independently verified all saved terminal receipt hashes and the empty native store. The CPU control reproduced the path failure, the corrected path reached the approved driver on all four lanes and rank failure propagated. | [Failure result](RESULT-NATIVE.md) and [prepared retry02 command](ALERT.md). Parent controls any new allocation. |
| 22:21 | Repeated the two controls and forced-failure check from the final detached recovery checkout. All three returned the expected status. The production store still contained only retry01. | Closed failure monitoring. The prepared command is in [ALERT](ALERT.md). |

At the initial check both detached source checkouts matched their expected commits and had clean tracked files. Git shows that only `launch.sh` changed between `0bfa5b087a2b5d5ba9d6ecebafd9db5288947186` and `ac28987f221e336cef2ea64dc16dfffc9f0e74c9`. The production validator requires exact equality between the batch contract's code SHA and `RETRY_PIN`. Use the producing SHA from the PFS receipts for CPU validation and readback.
