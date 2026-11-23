# Interim production receipt check

At 16:04:11 Shanghai on November 23, 2026, distinct allocation hosts had both reported publication for **4 policy collection keys and 69 motor-babble keys**. All 73 pairs agreed on transition and episode counts. Six other babble keys had starts from two hosts and fewer than two publication receipts. Two were retries of babble01, whose simulator import failed. The report counts one publication per owner per key. The local 15:39 speed snapshot predates both additional publishing attempts.

| Seed 0 policy collection key | Transitions | Episodes |
|---|---:|---:|
| none/0, shard001 | 4,341 | 32 |
| none/0, shard003 | 4,257 | 28 |
| rot_z/90, shard001 | 5,520 | 16 |
| rot_z/90, shard002 | 5,520 | 16 |

The collect owners were collect01 (job-3ohhwfrwt34h) and collect-seed0-01 (job-hr51yhgxiogv). The babble publishers were babble03 (job-rmvtf5i2znts) and babblecpu01 (job-0uwxvy2croqa). Both hosts entered the same logical units despite the shared lock protocol. On nominal policy shard001 the second job logged replay of all eight existing four-episode checkpoints. This case establishes duplicate publication and shows why two receipts need not represent two independent simulations.

The per-key events, exact line numbers, reservation identities and per-batch log lines are in [the interim snapshot](interim-receipts.json). Current file and checkpoint comparisons follow in the completed audit.

The first surviving-file check finished at 16:09:23. Concatenating the retained checkpoints and applying the native NumPy compression recipe reproduced the complete NPZ SHA-256 for all four collect keys. On nominal shard003 both jobs logged writes to batches 2–6, and on rotated shard002 both logged writes to batches 1 and 3. The latter's surviving archives identify the seed-0 lane's PID on batch 1 and the original lane's PID on batch 3. These are shared-final-path replacements with different writer PIDs. No historical copies or per-writer hashes appeared in the inspected files. [First readback](interim-files.json)
