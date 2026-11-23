# Reproduce the production receipt audit

Run the commands in Bash from this local `partition/audit/` directory. The SSH commands use stdin for code and inputs. The Linux processes open the production files for reading, use the run's existing Python environment for NumPy, hide CUDA and disable Python bytecode writes. They neither import the simulator nor run a policy.

## Regenerate the key ledger from the frozen evidence

`receipts.json` fixes the event cutoff at 2026-11-23T08:10:04.311394Z. `survivors.json` records the subsequent per-key observations. The following commands reuse those two files without rereading the live store. `cmp` compares the regenerated files with the distributed ledger and counts.

```bash
set -euo pipefail
{
  wc -c < summarize.py
  cat summarize.py
  jq -sc '.' receipts.json survivors.json
} | ssh -T -o BatchMode=yes baige-dev \
  "env CUDA_VISIBLE_DEVICES= python3 -IB -c 'import sys; b=sys.stdin.buffer; exec(b.read(int(b.readline())))'" \
  > replay-keys.tsv 2> replay-counts.json
cmp duplicate-keys.tsv replay-keys.tsv
cmp counts.json replay-counts.json
```

The TSV records one row per cross-host key including the six with fewer than two publications. Owners are ordered by recorded start time. Publication times and owner counts are empty for starts without a publication. The two SHA-256 columns identify the actual current files read at audit time. Empty under current metadata owner means the babble sidecar schema carries no writer identity.

## Repeat the read against live production

```bash
set -euo pipefail
root=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1
ssh -T -o BatchMode=yes baige-dev \
  "cd '$root' && env CUDA_VISIBLE_DEVICES= python3 -IB -" \
  < capture_receipts.py > new-receipts.json
bash inspect.sh new-receipts.json > new-survivors.json
```

`capture_receipts.py` enumerates all reservation directories under `reference-native/attempts`. It includes collect and babble JSONL lane events through its reader-clock cutoff and records source line numbers. Each source hash covers the full byte prefix actually read, including any unfinished line. Unfinished and post-cutoff events do not enter the count. The selected diagnostic and checkpoint log lines are from the same read sweep. The logs have no common snapshot barrier. `receipt_cutoff_utc` and `read_finished_utc` bound the sweep. Later receipts can increase the totals.

`inspect.sh` sends the selected receipt snapshot and the local `../../speed/active-snapshot.json` to the Baige CPU reader. `inspect_survivors.py` verifies the expected key against the current NPZ, sidecar and collect checkpoints. It records full file SHA-256 values and, for each NPZ field, its dtype, shape and SHA-256 of C-order array bytes. It uses the pinned collect episode generator's seed tuple, task rotation and four episodes per batch. Reassembly runs NumPy's native concatenation and compression recipe entirely in memory. The checkpoint ZIP member prefix contains the writer's PID from the temporary filename. `lane_pid_evidence` links each observed PID to an example final sidecar with host, rank and attempt. Each stat tuple contains device, inode, size, modification nanoseconds and change nanoseconds. A stable tuple applies to that key's own read window.

Full deployed source hashes, the observed Python and NumPy versions and the local speed-snapshot comparison appear in `survivors.json`. The production reservations contain full code, policy and upstream revisions. The 16:04 interim receipts and their later survivor readback remain separately timestamped. [Interim findings](INTERIM.md) · [Production results](RESULT.md)
